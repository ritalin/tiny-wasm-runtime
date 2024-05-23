use std::collections::{HashMap, LinkedList};

use anyhow::{bail, Result};
use tracing::{instrument, trace, Level};

use crate::binary::{instruction::Instruction, module::Module, types::{Block, BlockType, ExportDesc}};

use super::{store::{ExternalFuncInst, FuncInst, InternalFuncInst, MemoryInst, Store}, value::{Label, Value}, wasi::WasiSnapshotPreview1};

type ExtFn = Box<dyn FnMut(&mut Store, Vec<Value>) -> Result<Option<Value>>>;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub pc: usize,
    pub insts: Vec<Instruction>,
    pub locals: Vec<Value>,
    pub arity: usize,
    pub sp: usize,
    pub labels: LinkedList<Label>,
}

#[derive(Default)]
pub struct Runtime {
    pub store: Store,
    pub stack: LinkedList<Value>,
    pub call_stack: LinkedList<Frame>,
    pub current_frame: Option<Frame>,
    pub import_fns: HashMap<(String, String), ExtFn>,
    pub wasi_fns: Option<WasiSnapshotPreview1>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("store", &self.store)
            .field("stack", &self.stack)
            .field("call_stack", &self.call_stack)
            .field("current_frame", &self.current_frame)
            .field("wasi_fns", &self.wasi_fns)
        .finish()
    }
}

impl Runtime {
    pub fn instanciate(wasm: impl AsRef<[u8]>) -> Result<Self> {
        let module = Module::new(wasm.as_ref())?;
        let store = Store::new(module)?;

        Ok(Runtime {
            store,
            ..Default::default()
        })
    }

    pub fn instanciate_with_wasi(wasm: impl AsRef<[u8]>, wasi: WasiSnapshotPreview1) -> Result<Self> {
        let module = Module::new(wasm.as_ref())?;
        let store = Store::new(module)?;

        Ok(Runtime {
            store,
            wasi_fns: Some(wasi),
            ..Default::default()
        })
    }

    pub fn call(&mut self, name: impl Into<String> + std::fmt::Debug, args: Vec<Value>) -> Result<Option<Value>> {
        let name = name.into();

        let Some(export_fn) = self.store.exports.lookup.get(&name) else {
            bail!("Not found exported fn");
        };

        match export_fn.desc {
            ExportDesc::Func(index) => {
                self.call_with_index(index as usize, args)
            }
        }
    }

    #[tracing::instrument(skip(self) level=tracing::Level::TRACE)]
    pub fn call_with_index(&mut self, fn_index: usize, args: Vec<Value>) -> Result<Option<Value>> {
        trace!("(CAll/IN)");

        let Some(func) = self.store.fns.get(fn_index) else {
            bail!("Fundtion is not found");
        };

        match func {
            FuncInst::Internal(func) => {
                if args.len() != func.fn_type.params.len() {
                    bail!("Number of function args is mismatched (args: {}, passed: {})", func.fn_type.params.len(), args.len());
                }
                for arg in args {
                    self.stack.push_front(arg);
                }
        
                let (frame, next_stack) = make_frame(&mut self.stack, func);
                self.stack = next_stack;

                self.execute_inst_call(frame)
            }
            FuncInst::External(func) => {
                if args.len() != func.fn_type.params.len() {
                    bail!("Number of function args is mismatched");
                }
                for arg in args {
                    self.stack.push_front(arg);
                }
        
                self.execute_ext_call(func.clone())
            }
        }
    }

    fn execute(&mut self) -> Result<()> {
        trace!("(CALL/IN)");

        loop {
            let Some(frame) = self.current_frame.as_mut() else {
                break;
            };

            let Some(inst) = frame.insts.get(frame.pc) else {
                break;
            };

            frame.pc += 1;
            
            match inst {
                Instruction::End => {
                    let (next_stack, next_frame) = execute_inst_end(frame, &mut self.stack, &mut self.call_stack)?;

                    self.stack = next_stack;
                    self.current_frame = next_frame;

                    // self.stack = rewind_stack(&mut self.stack, sp, arity)?;

                    // // フレームを巻き戻す
                    // if let Some(next_frame) = self.call_stack.pop_front() {
                    //     self.current_frame = Some(next_frame);
                    // };
                }
                Instruction::LocalGet(index) => execute_inst_push_from_local(frame, &mut self.stack, *index)?,
                Instruction::LocalSet(index) => execute_inst_pop_to_local(frame, &mut self.stack, *index)?,
                Instruction::I32Const(value) => execute_inst_i32_const(frame, &mut self.stack, *value)?,
                Instruction::I32Add => execute_inst_add(frame, &mut self.stack)?,
                Instruction::I32Sub => execute_inst_sub(frame, &mut self.stack)?,
                Instruction::I32LtS => execute_inst_lt(frame, &mut self.stack)?,
                Instruction::Call(index) => {
                    let fn_index = (*index as usize).clone();
                    let count = get_func_arg_count(&self.store.fns, *index)?;
                    let (args, next_stack) = pop_args(&mut self.stack, count);

                    self.stack = next_stack;

                    if let Some(ret_value) = self.call_with_index(fn_index, args)? {
                        self.stack.push_front(ret_value);
                    }
                }
                Instruction::If(Block(block_type)) => {
                    let (next_stack, next_frame) = execute_inst_if(frame.clone(), &mut self.stack, block_type)?;

                    self.stack = next_stack;
                    self.current_frame = Some(next_frame);
                }                   
                Instruction::Return => {
                    let (next_stack, next_frame) = execute_inst_return(frame, &mut self.stack, &mut self.call_stack)?;

                    self.stack = next_stack;
                    self.current_frame = next_frame;
                }
                Instruction::I32Store { offset, .. } => {
                    execute_inst_i32_store(frame, &mut self.stack, &mut self.store.memories, *offset)?
                } 
            };
        }

        Ok(())
    }

    fn execute_inst_call(&mut self, next_frame: Frame) -> Result<Option<Value>> {
        if let Some(ref frame) = self.current_frame {
            self.call_stack.push_front(frame.clone());
        }
        self.current_frame = Some(next_frame.clone());

        let arity = next_frame.arity;

        match  self.execute() {
            Err(err) => {
                self.call_stack = LinkedList::default();
                self.stack = LinkedList::default();
                bail!("Failed to call: {err}");
            }
            Ok(_) if arity > 0 => {
                match self.stack.pop_front() {
                    None => bail!("Not found return value from internal call"),
                    Some(value) => Ok(Some(value))
                }
            }
            Ok(_) => Ok(None),
        }
    }

    fn execute_ext_call(&mut self, func: ExternalFuncInst) -> Result<Option<Value>> {
        let (args, next_stack) = pop_args(&mut self.stack, func.fn_type.params.len());
        self.stack = next_stack;

        match func.mod_name.as_str() {
            "wasi_snapshot_preview1" => {
                let Some(ref wasi) = self.wasi_fns else {
                    bail!("Unsupported WASI");
                };

                wasi.invoke(&mut self.store, &func.fn_name, args)
            }
            _ => {
                let Some(call) = self.import_fns.get_mut(&(func.mod_name.to_string(), func.fn_name.to_string())) else {
                    bail!("Ext function is not found");
                };
        
                call(&mut self.store, args)
            }
        }
        
    }

    pub fn add_import(&mut self, mod_name: impl Into<String>, fn_name: impl Into<String>, 
        call: impl FnMut(&mut Store, Vec<Value>) -> Result<Option<Value>> + 'static) -> Result<()> 
    {
        self.import_fns.insert((mod_name.into(), fn_name.into()), Box::new(call));

        Ok(())
    }
}

fn get_func_arg_count(fns: &[FuncInst], index: u32) -> Result<usize> {
    let Some(func) = fns.get(index as usize) else {
        bail!("Function is not found: {index}");
    };

    match func {
        FuncInst::Internal(InternalFuncInst { fn_type, .. }) |
        FuncInst::External(ExternalFuncInst { fn_type, .. }) => {
            Ok(fn_type.params.len())
        }
    }
}

fn pop_args(stack: &mut LinkedList<Value>, count: usize) -> (Vec<Value>, LinkedList<Value>) {
    let next_stack = stack.split_off(count);
    let args = stack.iter().map(Value::clone).rev().collect::<Vec<_>>();

    (args, next_stack)
}

fn make_frame(stack: &mut LinkedList<Value>, func: &InternalFuncInst) -> (Frame, LinkedList<Value>) {
    let (args, next_stack) = pop_args(stack, func.fn_type.params.len());

    let locals = func.code.locals.iter().map(|lc| match lc {
        crate::binary::types::ValueType::I32 => Value::I32(0),
        crate::binary::types::ValueType::I64 => Value::I64(0),
    });

    let frame = Frame { 
        pc: 0, 
        insts: func.code.body.clone(), 
        locals: args.into_iter().chain(locals).collect::<Vec<_>>(),
        arity: func.fn_type.returns.len(), 
        sp: next_stack.len(),
        labels: LinkedList::default(),
    };

    (frame, next_stack)
}

fn rewind_stack(stack: &mut LinkedList<Value>, sp: usize, arity: usize) -> Result<LinkedList<Value>> {
    let next_stack = match arity {
        0 => {
            // 戻り値なし
            stack.split_off(stack.len() - sp)
        }
        _ => {
            // 戻り値あり
            let Some(value) = stack.pop_front() else {
                bail!("Not found return value");
            };

            let mut next_stack = stack.split_off(stack.len() - sp);
            next_stack.push_front(value);
            next_stack
        }
    };

    Ok(next_stack)
}

#[instrument(level=Level::TRACE)]
fn execute_inst_push_from_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
    trace!("(CALL/IN)");

    let Some(value) = frame.locals.get(index as usize) else {
        bail!("Not found local var: {index}");
    };

    stack.push_front(value.clone());

    Ok(())
}

#[instrument(level=Level::TRACE)]
fn execute_inst_pop_to_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
    trace!("(CALL/IN)");

    let Some(lc) = frame.locals.get_mut(index as usize) else {
        bail!("Not found local var: {index}");
    };
    let Some(value) = stack.pop_front() else {
        bail!("Stack is empty")
    };

    *lc = value;
    Ok(())
}

#[instrument(level=Level::TRACE)]
fn execute_inst_i32_const(_frame: &mut Frame, stack: &mut LinkedList<Value>, value: i32) -> Result<()> {
    trace!("(CALL/IN)");

    stack.push_front(Value::I32(value));
    Ok(())
}

#[instrument(level=Level::TRACE)]
fn execute_inst_add(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    trace!("(CALL/IN)");

    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs + rhs);
    Ok(())
}

#[instrument(level=Level::TRACE)]
fn execute_inst_sub(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    trace!("(CALL/IN)");

    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs - rhs);
    Ok(())
}

fn execute_inst_lt(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front((lhs < rhs).into());
    Ok(())
}

fn execute_inst_i32_store(_frame: &mut Frame, stack: &mut LinkedList<Value>, memories: &mut [MemoryInst], offset: u32) -> Result<()> {
    let (Some(value), Some(addr)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("The i32.store value set is not found");
    };
    let Some(dest) = memories.get_mut(0) else {
        bail!("Memory is not found");
    };

    let addr = Into::<i32>::into(addr) as usize;
    let offset = offset as usize;
    let value = Into::<i32>::into(value);
    let sz = std::mem::size_of::<i32>();

    // copy_from_sliceはsrcとdstのサイズを合わせる必要がある
    dest.data[addr+offset..][..sz].copy_from_slice(&value.to_le_bytes());

    Ok(())
}

fn execute_inst_end(frame: &mut Frame, stack: &mut LinkedList<Value>, call_stack: &mut LinkedList<Frame>) -> Result<(LinkedList<Value>, Option<Frame>)> {
    match frame.labels.pop_front() {
        Some(Label { pc, sp, arity }) => {
            let next_stack = rewind_stack(stack, sp, arity)?;
            let mut next_frame = frame.clone();
            next_frame.pc = pc;

            Ok((next_stack, Some(next_frame)))
        }
        None => {
            let next_stack = rewind_stack(stack, frame.sp, frame.arity)?;
            Ok((next_stack, call_stack.pop_front()))
        }
    }
}

fn execute_inst_return(frame: &mut Frame, stack: &mut LinkedList<Value>, call_stack: &mut LinkedList<Frame>) -> Result<(LinkedList<Value>, Option<Frame>)> {
    let next_stack = rewind_stack(stack, frame.sp, frame.arity)?;
    Ok((next_stack, call_stack.pop_front()))
}

#[tracing::instrument(level=tracing::Level::TRACE)]
fn execute_inst_if(frame: Frame, stack: &mut LinkedList<Value>, block_type: &BlockType) -> Result<(LinkedList<Value>, Frame)> {
    trace!("execute_inst_if/IN");

    let mut frame = frame;
    
    let (end_addr, next_else) = get_end_addr(&frame);

    let Some(Value::I32(cond)) = stack.pop_front() else {
        bail!("Not found if condition");
    };

    match cond == 1 {
        false => match next_else {
            Some(addr) => frame.pc = addr,
            None => frame.pc = end_addr + 1, // end following
        }
        true => {
            let arity = match block_type {
                BlockType::Void => 0,
                BlockType::Value(v) => v.len(),
            };

            frame.labels.push_front(Label { pc: end_addr, arity, sp: stack.len() });
        }
    };

    Ok((stack.clone(), frame))
}

fn get_end_addr(frame: &Frame) -> (usize, Option<usize>) {
    let mut depth = 0;
    let mut addr = frame.pc;

    for inst in frame.insts.iter().skip(frame.pc) { // !!!!!
        match inst {
            Instruction::If(_) => { // TODO: treat other loop
                depth += 1;
            }
            Instruction::End if depth > 0 => {
                depth -= 1;
            }
            Instruction::End => {
                break;
            }
            _ => {}
        }

        addr += 1;
    }
    
    (addr, None) // TODO: treat else inst
}

#[cfg(test)]
mod executor_tests {
    use std::collections::LinkedList;

    use anyhow::Result;
    use crate::{binary::{
        instruction::Instruction, module::Module, 
        types::{Block, BlockType, FuncType, ValueType}}, 
        execution::{runtime::Runtime, 
            store::{ExternalFuncInst, FuncInst, InternalFuncInst, MemoryInst, Store, PPAGE_SIZE}, 
            value::{Label, Value}
        }
    };

    use super::Frame;

    #[test]
    fn execute_simplest_fn() -> Result<()> {
        let wasm = wat::parse_str("(module (func))")?;

        let mut rt = Runtime::instanciate(wasm)?;

        rt.execute()?;

        assert_eq!(0, rt.call_stack.len());
        assert_eq!(0, rt.stack.len());
        Ok(())
    }

    #[test]
    fn execute_fn_add() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(5)), instance.call_with_index(0, vec![Value::I32(2), Value::I32(3)])?);
        assert_eq!(Some(Value::I32(15)), instance.call_with_index(0, vec![Value::I32(10), Value::I32(5)])?);
        assert_eq!(Some(Value::I32(2)), instance.call_with_index(0, vec![Value::I32(1), Value::I32(1)])?);
        Ok(())
    }

    #[test]
    fn execute_fn_sub() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.sub))")?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(42)), instance.call_with_index(0, vec![Value::I32(60), Value::I32(18)])?);
        assert_eq!(Some(Value::I32(-21)), instance.call_with_index(0, vec![Value::I32(21), Value::I32(42)])?);
        assert_eq!(Some(Value::I32(20)), instance.call_with_index(0, vec![Value::I32(20), Value::I32(0)])?);
        Ok(())
    }

    #[test]
    fn execute_fn_le() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func $fib (param i32 i32)(result i32) (i32.lt_s (local.get 0) (local.get 1))))"#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(0)), instance.call_with_index(0, vec![Value::I32(60), Value::I32(18)])?);
        assert_eq!(Some(Value::I32(1)), instance.call_with_index(0, vec![Value::I32(18), Value::I32(60)])?);
        assert_eq!(Some(Value::I32(0)), instance.call_with_index(0, vec![Value::I32(42), Value::I32(42)])?);
        Ok(())
    }

    #[test]
    fn execute_fn_add_by_name() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (export "add" (func $add)) (func $add (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))"#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(5)), instance.call("add", vec![Value::I32(2), Value::I32(3)])?);
        assert_eq!(Some(Value::I32(15)), instance.call("add", vec![Value::I32(10), Value::I32(5)])?);
        assert_eq!(Some(Value::I32(2)), instance.call("add", vec![Value::I32(1), Value::I32(1)])?);
        Ok(())
    }

    #[test]
    fn execute_other_fn() -> Result<()> {
        let source = r#"
            (module 
                (func (export "call_doubler") (param i32) (result i32) (local.get 0) (call $double))
                (func $double (param i32) (result i32) (local.get 0) (local.get 0) i32.add)
            )
        "#;
        let wasm = wat::parse_str(source)?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(42)), instance.call("call_doubler", vec![Value::I32(21)])?);
        Ok(())
    }

    #[test]
    fn execute_import_fn() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func $double (import "env" "double_ext") (param i32) (result i32)))"#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        instance.add_import("env", "double_ext", |_, args| Ok(Some(args[0] + args[0])))?;

        assert_eq!(Some(Value::I32(198)), instance.call_with_index(0, vec![Value::I32(99)])?);
        Ok(())
    }

    #[test]
    fn execute_pop_to_local() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func (result i32) (local $x i32) (local.set $x (i32.const -42)) (local.get 0)))"#)?;
        let mut instance = Runtime::instanciate(wasm)?;
        
        assert_eq!(Some(Value::I32(-42)), instance.call_with_index(0, vec![])?);
        Ok(())
    }

    #[test]
    fn execute_i32_store() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (memory 1) (func (i32.const 10) (i32.const 42) (i32.store)))"#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        _ = instance.call_with_index(0, vec![])?;

        assert_eq!(42, instance.store.memories[0].data[10]);
        Ok(())
    }

    #[test]
    fn execute_return_fn() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func (result i32) (i32.const 11) (i32.const 22) (return (i32.const 33))))"#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(33)), instance.call_with_index(0, vec![])?);

        assert_eq!(0, instance.stack.len());
        Ok(())
    }

    #[test]
    fn execute_if_true() -> Result<()> {
        let wasm = wat::parse_str(r#"
        (module (func (result i32) 
            (i32.const 1) 
            (if 
                (then (return (i32.const 22)))
            )
            (return (i32.const 33))
        ))
        "#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(22)), instance.call_with_index(0, vec![])?);

        assert_eq!(0, instance.stack.len());
        Ok(())
    }

    #[test]
    fn execute_if_false() -> Result<()> {
        let wasm = wat::parse_str(r#"
        (module (func (result i32) 
            (i32.const 0) 
            (if 
                (then (return (i32.const 22)))
            )
            (return (i32.const 33))
        ))
        "#)?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(33)), instance.call_with_index(0, vec![])?);

        assert_eq!(0, instance.stack.len());
        Ok(())        
    }

    #[test]
    fn init_store() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        assert_eq!(1, store.fns.len());

        let expect = InternalFuncInst {
            fn_type: FuncType { params: vec![ValueType::I32, ValueType::I32], returns: vec![ValueType::I32] },
            code: crate::execution::store::Function { 
                locals: vec![], 
                body: vec![
                    Instruction::LocalGet(0),
                    Instruction::LocalGet(1),
                    Instruction::I32Add,
                    Instruction::End,
                ],
            },
        };
        assert_eq!(FuncInst::Internal(expect), store.fns[0]);
        Ok(())
    }

    #[test]
    fn init_store_with_ext_fn() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func $dummy (import "env" "dummy")(param i32 i32)(result i32)))"#)?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        assert_eq!(1, store.fns.len());

        let expect = ExternalFuncInst {
                fn_type: FuncType { params: vec![ValueType::I32, ValueType::I32], returns: vec![ValueType::I32] },
                mod_name: "env".to_string(),
                fn_name: "dummy".to_string()
            }
        ;
        assert_eq!(FuncInst::External(expect), store.fns[0]);
        Ok(())
    }

    #[test]
    fn init_store_memory() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (memory 2) (data (i32.const 42)) (data (i32.const -42)))"#)?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        assert_eq!(65535, PPAGE_SIZE);
        assert_eq!(1, store.memories.len());
        assert_eq!(2 * PPAGE_SIZE, store.memories[0].data.len());
        assert_eq!(None, store.memories[0].limit);
        Ok(())
    }

    #[test]
    fn init_store_const_data() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (memory 1) (data (i32.const 0) "Hello") (data (i32.const 5) "World\n"))"#)?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        assert_eq!(b"HelloWorld\n", &store.memories[0].data[0..11]);
        Ok(())
    }

    #[test]
    fn eval_inst_push_from_locals() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(2)],
            ..Default::default()
        };
        let mut stack = LinkedList::<Value>::new();

        super::execute_inst_push_from_local(&mut frame, &mut stack, 0)?;

        assert_eq!(1, stack.len());
        assert_eq!(Some(Value::I32(2)), stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_add_i32() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(5), Value::I32(10)],
            ..Default::default()
        };
        let mut stack = LinkedList::<Value>::new();

        assert_eq!(false, super::execute_inst_add(&mut frame, &mut stack).is_ok());

        super::execute_inst_push_from_local(&mut frame, &mut stack, 0)?;
        super::execute_inst_push_from_local(&mut frame, &mut stack, 1)?;
        assert_eq!(true, super::execute_inst_add(&mut frame, &mut stack).is_ok());

        assert_eq!(1, stack.len());
        assert_eq!(Some(Value::I32(15)), stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_i32_sub() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![],
            ..Default::default()
        };

        let mut stack = LinkedList::<Value>::from([Value::I32(10), Value::I32(5)]);
        assert_eq!(true, super::execute_inst_sub(&mut frame, &mut stack).is_ok());

        assert_eq!(1, stack.len());
        assert_eq!(Some(Value::I32(-5)), stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_i32_lt_s() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![],
            ..Default::default()
        };

        let mut stack = LinkedList::<Value>::from([Value::I32(10), Value::I32(5)]);
        assert_eq!(true, super::execute_inst_lt(&mut frame, &mut stack).is_ok());

        assert_eq!(1, stack.len());
        assert_eq!(Some(Value::I32(1)), stack.front().map(|v| v.clone()));        
        Ok(())
    }

    #[test]
    fn make_frame_of_without_return() -> Result<()> {
        let wasm = wat::parse_str("(module (func))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        let mut stack = LinkedList::<Value>::new();

        let Some(FuncInst::Internal(fn_decl)) = store.fns.get(0) else {
            unreachable!();
        };

        let expect = Frame { 
            insts: vec![
                Instruction::End,
            ], 
            ..Default::default() 
        };

        let (frame, next_stack) = super::make_frame(&mut stack, fn_decl);
        assert_eq!(expect, frame);
        assert_eq!(0, next_stack.len());

        Ok(())
    }

    #[test]
    fn make_frame_of_with_return() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32) (result i32) (local.get 0)))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        let mut stack = LinkedList::from([Value::I32(42)]);

        let Some(FuncInst::Internal(fn_decl)) = store.fns.get(0) else {
            unreachable!();
        };

        let expect = Frame { 
            insts: vec![
                Instruction::LocalGet(0),
                Instruction::End,
            ], 
            locals: vec![Value::I32(42)],
            arity: 1,
            ..Default::default() 
        };

        let (frame, next_stack) = super::make_frame(&mut stack, fn_decl);
        assert_eq!(expect, frame);
        assert_eq!(0, next_stack.len());

        Ok(())
    }

    #[test]
    fn make_frame_fn_add() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        let mut stack = LinkedList::from([Value::I32(10), Value::I32(5)]);

        let Some(FuncInst::Internal(fn_decl)) = store.fns.get(0) else {
            unreachable!();
        };

        let expect = Frame { 
            insts: vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Add,
                Instruction::End,
            ], 
            locals: vec![Value::I32(5), Value::I32(10)],
            arity: 1,
            ..Default::default() 
        };

        let (frame, next_stack) = super::make_frame(&mut stack, fn_decl);
        assert_eq!(expect, frame);
        assert_eq!(0, next_stack.len());

        Ok(())
    }

    #[test]
    fn eval_inst_call_noreturn() -> Result<()> {
        let next_frame = Frame { 
            locals: vec![Value::I32(42)], 
            insts: vec![
                Instruction::End,
            ], 
            ..Default::default() 
        };

        let mut rt = Runtime { store: Store::new(Module::default())?, ..Default::default() };
        let ret_value = rt.execute_inst_call(next_frame)?;

        assert_eq!(None, ret_value);
        assert_eq!(0, rt.call_stack.len());
        assert_eq!(0, rt.stack.len());
        Ok(())    
    }

    #[test]
    fn eval_inst_call() -> Result<()> {
        let next_frame = Frame { 
            locals: vec![Value::I32(42)], 
            insts: vec![
                Instruction::LocalGet(0),
                Instruction::End,
            ], 
            arity: 1, 
            ..Default::default() 
        };

        let mut rt = Runtime { store: Store::new(Module::default())?, ..Default::default() };
        let ret_value = rt.execute_inst_call(next_frame)?;

        assert_eq!(Some(Value::I32(42)), ret_value);
        assert_eq!(0, rt.call_stack.len());
        assert_eq!(0, rt.stack.len());
        Ok(())
    }

    #[test]
    fn eval_inst_local_set() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(0), ],
            ..Default::default()
        };
        let mut stack = LinkedList::<Value>::from([Value::I32(42)]);

        super::execute_inst_pop_to_local(&mut frame, &mut stack, 0)?;

        assert_eq!(0, stack.len());
        assert_eq!(Value::I32(42), frame.locals[0]);
        Ok(())
    }

    #[test]
    fn eval_inst_i32_const() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![],
            ..Default::default()
        };
        let mut stack = LinkedList::<Value>::new();

        super::execute_inst_i32_const(&mut frame, &mut stack, -123)?;

        assert_eq!(1, stack.len());
        assert_eq!(Some(Value::I32(-123)), stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_i32_store() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(42), Value::I32(0)]);
        let mut memories = vec![MemoryInst { data: vec![0; 100], limit: None }];

        let mut frame = Frame::default();

        super::execute_inst_i32_store(&mut frame, &mut stack, &mut memories, 10)?;

        assert_eq!(42, memories[0].data[10]);
        Ok(())
    }

    #[test]
    fn eval_inst_end_from_fn_noreturn() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        // Labelが積まれていない状態でEND命令を評価する場合、関数からの脱出とみなす
        let mut frame = Frame { 
            labels: LinkedList::default(),
            arity: 0, sp: 0, 
            ..Default::default()
        };
        let mut call_stack = LinkedList::<Frame>::from([Frame::default()]);

        let (next_stack, next_frame) = super::execute_inst_end(&mut frame, &mut stack, &mut call_stack)?;
        
        assert_eq!(0, next_stack.len());
        assert_eq!(0, call_stack.len());
        assert_eq!(Some(Frame::default()), next_frame);

        Ok(())
    }

    #[test]
    fn eval_inst_end_from_block() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let mut frame = Frame { 
            labels: LinkedList::from([
                Label { pc: 13, arity: 1, sp: 1 }, 
                Label::default(), 
            ]), 
            arity: 0, sp: 0, 
            ..Default::default()
        };
        let mut call_stack = LinkedList::<Frame>::from([Frame::default()]);

        let (next_stack, next_frame) = super::execute_inst_end(&mut frame, &mut stack, &mut call_stack)?;
        
        assert_eq!(2, next_stack.len());
        assert_eq!(vec![Value::I32(42), Value::I32(44)], next_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(1, call_stack.len());
        assert_eq!(vec![Frame::default()], call_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(Some(Frame { labels: LinkedList::from([Label::default()]), pc: 13, arity: 0, sp: 0, ..Default::default() }), next_frame);

        Ok(())
    }

    #[test]
    fn eval_inst_end_from_block_noreturn() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let mut frame = Frame { 
            labels: LinkedList::from([
                Label { pc: 99, arity: 0, sp: 1, }, 
                Label::default(), 
            ]), 
            arity: 1, sp: 0, 
            ..Default::default() 
        };
        let mut call_stack = LinkedList::<Frame>::from([Frame::default()]);

        let (next_stack, next_frame) = super::execute_inst_end(&mut frame, &mut stack, &mut call_stack)?;
        
        assert_eq!(1, next_stack.len());
        assert_eq!(vec![Value::I32(44)], next_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(1, call_stack.len());
        assert_eq!(vec![Frame::default()], call_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(Some(Frame { labels: LinkedList::from([Label::default()]), pc: 99, arity: 1, sp: 0, ..Default::default() }), next_frame);
        Ok(())
    }

    #[test]
    fn eval_inst_return_from_fn() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let mut frame = Frame { 
            arity: 1, sp: 0, 
            ..Default::default() 
        };
        let mut call_stack = LinkedList::<Frame>::from([Frame::default()]);

        let (next_stack, next_frame) = super::execute_inst_return(&mut frame, &mut stack, &mut call_stack)?;
        
        assert_eq!(1, next_stack.len());
        assert_eq!(vec![Value::I32(42)], next_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(0, call_stack.len());
        assert_eq!(Some(Frame::default()), next_frame);
        Ok(())
    }

    #[test]
    fn eval_inst_return_void_from_fn() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let mut frame = Frame { 
            arity: 0, sp: 0, 
            ..Default::default() 
        };
        let mut call_stack = LinkedList::<Frame>::from([Frame::default()]);

        let (next_stack, next_frame) = super::execute_inst_return(&mut frame, &mut stack, &mut call_stack)?;
        
        assert_eq!(0, next_stack.len());
        assert_eq!(0, call_stack.len());
        assert_eq!(Some(Frame::default()), next_frame);
        Ok(())
    }

    #[test]
    fn eval_inst_if_true() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(1)]);
        let frame = Frame{ 
            pc: 2,
            insts: vec![
                Instruction::I32Const(1),
                Instruction::If(Block(BlockType::Void)),
                Instruction::I32Const(20),
                Instruction::If(Block(BlockType::Void)),
                Instruction::I32Const(30),
                Instruction::End,
                Instruction::End,
                Instruction::I32Const(40),
            ],
            ..Default::default() 
        };

        let (next_stack, next_frame) = super::execute_inst_if(frame, &mut stack, &BlockType::Void)?;

        assert_eq!(0, next_stack.len());

        assert_eq!(2, next_frame.pc);
        assert_eq!(0, next_frame.sp);
        assert_eq!(0, next_frame.arity);
        assert_eq!(1, next_frame.labels.len());
        assert_eq!(Some(&Label { pc: 6, sp: 0, arity: 0 }), next_frame.labels.front());
        Ok(())
    }

    #[test]
    fn eval_inst_if_false() -> Result<()> {
        let mut stack = LinkedList::<Value>::from([Value::I32(0)]);
        let frame = Frame{ 
            pc: 2,
            insts: vec![
                Instruction::I32Const(0),
                Instruction::If(Block(BlockType::Void)),
                Instruction::I32Const(20),
                Instruction::If(Block(BlockType::Void)),
                Instruction::I32Const(30),
                Instruction::End,
                Instruction::End,
                Instruction::I32Const(40),
            ],
            ..Default::default() 
        };

        let (next_stack, next_frame) = super::execute_inst_if(frame, &mut stack, &BlockType::Void)?;

        assert_eq!(0, next_stack.len());

        assert_eq!(7, next_frame.pc);
        assert_eq!(0, next_frame.sp);
        assert_eq!(0, next_frame.arity);
        assert_eq!(0, next_frame.labels.len());
        Ok(())
    }
}

#[cfg(test)]
mod wasi_test {
    use anyhow::Result;

    use crate::execution::wasi::WasiSnapshotPreview1;

    use super::Runtime;

    #[test]
    fn invoke_wasi() -> Result<()> {
        let wasm = wat::parse_file("src/fixtures/invoke_wasi.wat")?;
        let wasi = WasiSnapshotPreview1 {};

        let mut rt = Runtime::instanciate_with_wasi(wasm, wasi)?;
        let _ = rt.call("_start", vec![]);
        Ok(())
    }
}