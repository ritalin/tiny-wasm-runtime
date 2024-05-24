use std::collections::{HashMap, LinkedList};

use anyhow::{bail, Result};
use tracing::trace;

use crate::binary::{module::Module, types::{Instruction, Block, ExportDesc}};

use super::{
    store::{ExternalFuncInst, FuncInst, InternalFuncInst, Store}, 
    value::{Label, Value},
    instruction::*,
    wasi::WasiSnapshotPreview1
};

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

#[cfg(test)]
mod executor_tests {
    use anyhow::Result;
    use crate::{binary::{module::Module, types::Instruction}, execution::{runtime::{Frame, Runtime}, store::Store, value::Value
        }};

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