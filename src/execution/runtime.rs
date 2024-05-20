use std::collections::{HashMap, LinkedList};

use anyhow::{bail, Result};

use crate::binary::{instruction::Instruction, module::Module, types::ExportDesc};

use super::{store::{ExternalFuncInst, FuncInst, InternalFuncInst, Store}, value::Value};

type ExtFn = Box<dyn FnMut(&mut Store, Vec<Value>) -> Result<Option<Value>>>;

#[derive(Default, Debug, PartialEq, Eq)]
pub struct Frame {
    pub pc: usize,
    pub insts: Vec<Instruction>,
    pub locals: Vec<Value>,
    pub arity: usize,
    pub sp: usize,
}

#[derive(Default)]
pub struct Runtime {
    pub store: Store,
    pub stack: LinkedList<Value>,
    pub call_stack: LinkedList<Frame>,
    pub import_fns: HashMap<(String, String), ExtFn>,
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

    pub fn call(&mut self, name: impl Into<String>, args: Vec<Value>) -> Result<Option<Value>> {
        let Some(export_fn) = self.store.exports.lookup.get(&name.into()) else {
            bail!("Not found exported fn");
        };

        match export_fn.desc {
            ExportDesc::Func(index) => {
                self.call_with_index(index as usize, args)
            }
        }
    }

    pub fn call_with_index(&mut self, fn_index: usize, args: Vec<Value>) -> Result<Option<Value>> {
        let Some(func) = self.store.fns.get(fn_index) else {
            bail!("Fundtion is not found");
        };

        for arg in args {
            self.stack.push_front(arg);
        }

        match func {
            FuncInst::Internal(func) => {
                let (frame, next_stack) = make_frame(&mut self.stack, func);
                self.stack = next_stack;

                self.execute_inst_call(frame)
            }
            FuncInst::External(func) => {
                self.execute_ext_call(func.clone())
            }
        }
    }

    fn execute(&mut self) -> Result<()> {
        loop {
            let Some(frame) = self.call_stack.front_mut() else {
                break;
            };

            let Some(inst) = frame.insts.get(frame.pc) else {
                break;
            };

            frame.pc += 1;

            match inst {
                Instruction::End => {
                    let Some(Frame { sp, arity, .. }) = self.call_stack.pop_front() else {
                        bail!("Not found frame");
                    };

                    match arity {
                        0 => {
                            // 戻り値なし
                            self.stack = self.stack.split_off(sp);
                        }
                        _ => {
                            // 戻り値あり
                            let Some(value) = self.stack.pop_front() else {
                                bail!("Not found return value");
                            };

                            self.stack = self.stack.split_off(sp);
                            self.stack.push_front(value);
                        }
                    }
                }
                Instruction::LocalGet(index) => execute_inst_push_from_local(frame, &mut self.stack, *index)?,
                Instruction::LocalSet(index) => execute_inst_pop_to_local(frame, &mut self.stack, *index)?,
                Instruction::I32Const(value) => execute_inst_i32_const(frame, &mut self.stack, *value)?,
                Instruction::I32Add => execute_inst_add(frame, &mut self.stack)?,
                Instruction::Call(index) => {
                    let fn_index = (*index as usize).clone();

                    if let Some(ret_value) = self.call_with_index(fn_index, vec![])? {
                        self.stack.push_front(ret_value);
                    }
                }
                _ => todo!()
            };
        }

        Ok(())
    }

    fn execute_inst_call(&mut self, next_frame: Frame) -> Result<Option<Value>> {
        let arity = next_frame.arity;
        self.call_stack.push_front(next_frame);

        match  self.execute() {
            Err(err) => {
                self.call_stack = LinkedList::default();
                self.stack = LinkedList::default();
                bail!("Failed to call: {err}");
            }
            Ok(_) if arity > 0 => {
                match self.stack.pop_front() {
                    None => bail!("Not found return value"),
                    Some(value) => Ok(Some(value))
                }
            }
            Ok(_) => Ok(None),
        }
    }

    fn execute_ext_call(&mut self, func: ExternalFuncInst) -> Result<Option<Value>> {
        let Some(call) = self.import_fns.get_mut(&(func.mod_name.to_string(), func.fn_name.to_string())) else {
            bail!("Ext function is not found");
        };

        let (args, next_stack) = pop_args(&mut self.stack, func.fn_type.params.len());
        self.stack = next_stack;

        call(&mut self.store, args)
    }

    pub fn add_import(&mut self, mod_name: impl Into<String>, fn_name: impl Into<String>, 
        call: impl FnMut(&mut Store, Vec<Value>) -> Result<Option<Value>> + 'static) -> Result<()> 
    {
        self.import_fns.insert((mod_name.into(), fn_name.into()), Box::new(call));

        Ok(())
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
        sp: next_stack.len() 
    };

    (frame, next_stack)
}

fn execute_inst_push_from_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
    let Some(value) = frame.locals.get(index as usize) else {
        bail!("Not found local var: {index}");
    };

    stack.push_front(value.clone());
    Ok(())
}

fn execute_inst_pop_to_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
    let Some(lc) = frame.locals.get_mut(index as usize) else {
        bail!("Not found local var: {index}");
    };
    let Some(value) = stack.pop_front() else {
        bail!("Stack is empty")
    };

    *lc = value;
    Ok(())
}

fn execute_inst_i32_const(_frame: &mut Frame, stack: &mut LinkedList<Value>, value: i32) -> Result<()> {
    stack.push_front(Value::I32(value));
    Ok(())
}

fn execute_inst_add(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs + rhs);
    Ok(())
}

#[cfg(test)]
mod executor_tests {
    use std::collections::LinkedList;

    use anyhow::Result;
    use crate::{binary::{
        instruction::Instruction, module::Module, 
        types::{FuncType, ValueType}}, 
        execution::{runtime::Runtime, 
            store::{ExternalFuncInst, FuncInst, InternalFuncInst, Store, PPAGE_SIZE}, 
            value::Value
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
        let wasm = wat::parse_bytes(source.as_bytes())?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(42)), instance.call("call_doubler", vec![Value::I32(21)])?);
        Ok(())
    }

    #[test]
    fn execute_import_fn() -> Result<()> {
        let wasm = wat::parse_bytes(r#"(module (func $double (import "env" "double_ext") (param i32) (result i32)))"#.as_bytes())?;
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
        todo!("13章で実装予定")        
    }
}