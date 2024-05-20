use std::{collections::LinkedList};

use anyhow::{bail, Result};

use crate::binary::{instruction::Instruction, module::Module};

use super::{store::{FuncInst, InternalFuncInst, Store}, value::Value};

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

    pub fn call(&mut self, fn_index: usize, args: Vec<Value>) -> Result<Option<Value>> {
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
                Instruction::LocalGet(index) => execute_inst_push_local(frame, &mut self.stack, *index)?,
                Instruction::I32Add => execute_inst_add(frame, &mut self.stack)?,
            };
        }

        Ok(())
    }

    fn execute_inst_call(&mut self, next_frame: Frame) -> Result<Option<Value>> {
        let arity = next_frame.arity;
        self.call_stack.push_back(next_frame);

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
}

fn make_frame(stack: &mut LinkedList<Value>, func: &InternalFuncInst) -> (Frame, LinkedList<Value>) {
    let next_stack = stack.split_off(func.fn_type.params.len());
    let locals = stack.iter().map(Value::clone).rev().collect::<Vec<_>>();

    let frame = Frame { 
        pc: 0, 
        insts: func.code.body.clone(), 
        locals,
        arity: func.fn_type.returns.len(), 
        sp: next_stack.len() 
    };

    (frame, next_stack)
}

fn execute_inst_push_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
    let Some(value) = frame.locals.get(index as usize) else {
        bail!("Not found local var: {index}");
    };

    stack.push_front(value.clone());
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
            store::{FuncInst, InternalFuncInst, Store}, 
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

        assert_eq!(Some(Value::I32(5)), instance.call(0, vec![Value::I32(2), Value::I32(3)])?);
        assert_eq!(Some(Value::I32(15)), instance.call(0, vec![Value::I32(10), Value::I32(5)])?);
        assert_eq!(Some(Value::I32(2)), instance.call(0, vec![Value::I32(1), Value::I32(1)])?);
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
            }
        ;
        assert_eq!(FuncInst::Internal(expect), store.fns[0]);
        Ok(())
    }

    #[test]
    fn eval_inst_push_locals() -> Result<()> {
        let mut frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(2)],
            ..Default::default()
        };
        let mut stack = LinkedList::<Value>::new();

        super::execute_inst_push_local(&mut frame, &mut stack, 0)?;

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

        super::execute_inst_push_local(&mut frame, &mut stack, 0)?;
        super::execute_inst_push_local(&mut frame, &mut stack, 1)?;
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
}