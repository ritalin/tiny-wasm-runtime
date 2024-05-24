use std::collections::LinkedList;
use anyhow::{bail, Result};
use tracing::{instrument, trace, Level};

use crate::binary::types::{BlockType, Instruction};

use super::{runtime::Frame, store::{InternalFuncInst, MemoryInst}, value::{Label, Value}};

pub fn pop_args(stack: &mut LinkedList<Value>, count: usize) -> (Vec<Value>, LinkedList<Value>) {
    let next_stack = stack.split_off(count);
    let args = stack.iter().map(Value::clone).rev().collect::<Vec<_>>();

    (args, next_stack)
}

pub fn make_frame(stack: &mut LinkedList<Value>, func: &InternalFuncInst) -> (Frame, LinkedList<Value>) {
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

#[instrument(level=Level::TRACE)]
pub fn execute_inst_push_from_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
    trace!("(CALL/IN)");

    let Some(value) = frame.locals.get(index as usize) else {
        bail!("Not found local var: {index}");
    };

    stack.push_front(value.clone());

    Ok(())
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_pop_to_local(frame: &mut Frame, stack: &mut LinkedList<Value>, index: u32) -> Result<()> {
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
pub fn execute_inst_i32_const(_frame: &mut Frame, stack: &mut LinkedList<Value>, value: i32) -> Result<()> {
    trace!("(CALL/IN)");

    stack.push_front(Value::I32(value));
    Ok(())
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_add(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    trace!("(CALL/IN)");

    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs + rhs);
    Ok(())
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_sub(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    trace!("(CALL/IN)");

    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs - rhs);
    Ok(())
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_lt(_frame: &mut Frame, stack: &mut LinkedList<Value>) -> Result<()> {
    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front((lhs < rhs).into());
    Ok(())
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_i32_store(_frame: &mut Frame, stack: &mut LinkedList<Value>, memories: &mut [MemoryInst], offset: u32) -> Result<()> {
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

#[instrument(level=Level::TRACE)]
pub fn execute_inst_end(frame: &mut Frame, stack: &mut LinkedList<Value>, call_stack: &mut LinkedList<Frame>) -> Result<(LinkedList<Value>, Option<Frame>)> {
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

#[instrument(level=Level::TRACE)]
pub fn execute_inst_return(frame: &mut Frame, stack: &mut LinkedList<Value>, call_stack: &mut LinkedList<Frame>) -> Result<(LinkedList<Value>, Option<Frame>)> {
    let next_stack = rewind_stack(stack, frame.sp, frame.arity)?;
    Ok((next_stack, call_stack.pop_front()))
}

#[tracing::instrument(level=tracing::Level::TRACE)]
pub fn execute_inst_if(frame: Frame, stack: &mut LinkedList<Value>, block_type: &BlockType) -> Result<(LinkedList<Value>, Frame)> {
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

#[cfg(test)]
mod inst_tests {
    use std::collections::LinkedList;

    use anyhow::Result;

    use crate::{binary::{module::Module, types::{Block, BlockType, Instruction}}, execution::{runtime::Frame, store::{FuncInst, MemoryInst, Store}, value::{Label, Value}}};

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

        let insts = vec![
                Instruction::LocalGet(0),
                Instruction::End,
            ];
        let expect = Frame { 
            insts, 
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