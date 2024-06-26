use std::collections::LinkedList;
use anyhow::{bail, Result};
use tracing::{instrument, trace, Level};

use crate::binary::types::{BlockType, Instruction};
use super::{store::{InternalFuncInst, MemoryInst}, value::{Label, Value}};

pub type Stack = LinkedList<Value>;
pub type CallStack = LinkedList<Frame>;

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
pub struct EvalResultContext {
    // 関数内の命令がまだ続く場合、Some(Frame)を、消化し切った場合はNoneを割り当てる
    pub next_frame: Option<Frame>,
    pub next_stack: Stack
}

pub fn pop_args(mut stack: Stack, count: usize) -> (Vec<Value>, Stack) {
    let next_stack = stack.split_off(count);
    let args = stack.iter().map(Value::clone).rev().collect::<Vec<_>>();

    (args, next_stack)
}

pub fn make_frame(stack: Stack, func: &InternalFuncInst) -> (Frame, Stack) {
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
pub fn execute_inst_push_from_local(frame: Frame, mut stack: Stack, index: u32) -> Result<EvalResultContext> {
    trace!("(CALL/IN)");

    let Some(value) = frame.locals.get(index as usize) else {
        bail!("Not found local var: {index}");
    };

    stack.push_front(value.clone());

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_pop_to_local(mut frame: Frame, mut stack: Stack, index: u32) -> Result<EvalResultContext> {
    trace!("(CALL/IN)");

    let Some(lc) = frame.locals.get_mut(index as usize) else {
        bail!("Not found local var: {index}");
    };
    let Some(value) = stack.pop_front() else {
        bail!("Stack is empty")
    };

    *lc = value;

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_i32_const(frame: Frame, mut stack: Stack, value: i32) -> Result<EvalResultContext> {
    trace!("(CALL/IN)");

    stack.push_front(Value::I32(value));

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_add(frame: Frame, mut stack: Stack) -> Result<EvalResultContext> {
    trace!("(CALL/IN)");

    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs + rhs);

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_sub(frame: Frame, mut stack: Stack) -> Result<EvalResultContext> {
    trace!("(CALL/IN)");

    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front(lhs - rhs);

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_lt(frame: Frame, mut stack: Stack) -> Result<EvalResultContext> {
    let (Some(rhs), Some(lhs)) = (stack.pop_front(), stack.pop_front()) else {
        bail!("Not found enough value in stack");
    };

    stack.push_front((lhs < rhs).into());

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_i32_store(frame: Frame, mut stack: Stack, memories: &mut [MemoryInst], offset: u32) -> Result<EvalResultContext> {
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

    Ok(EvalResultContext { next_frame: Some(frame), next_stack: stack })
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_end(mut frame: Frame, stack: Stack) -> Result<EvalResultContext> {
    match frame.labels.pop_front() {
        Some(Label { pc, sp, arity }) => {
            let next_stack = rewind_stack(stack, sp, arity)?;
            
            frame.pc = pc;

            Ok(EvalResultContext { next_frame: Some(frame), next_stack: next_stack })
        }
        None => {
            let next_stack = rewind_stack(stack, frame.sp, frame.arity)?;
            Ok(EvalResultContext { next_frame: None, next_stack: next_stack })
        }
    }
}

#[instrument(level=Level::TRACE)]
pub fn execute_inst_return(frame: Frame, stack: Stack) -> Result<EvalResultContext> {
    let next_stack = rewind_stack(stack, frame.sp, frame.arity)?;

    Ok(EvalResultContext { next_frame: None, next_stack: next_stack })
}

#[tracing::instrument(level=tracing::Level::TRACE)]
pub fn execute_inst_if(mut frame: Frame, mut stack: Stack, block_type: &BlockType) -> Result<EvalResultContext> {
    trace!("execute_inst_if/IN");

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

    Ok(EvalResultContext {next_frame: Some(frame), next_stack: stack.clone() })
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

pub fn rewind_stack(mut stack: Stack, sp: usize, arity: usize) -> Result<Stack> {
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

    use crate::{
        binary::{module::Module, types::{Block, BlockType, Instruction}}, 
        execution::{
            instruction::{EvalResultContext, Frame, Stack}, store::{FuncInst, MemoryInst, Store}, 
            value::{Label, Value}
        }
    };

    #[test]
    fn eval_inst_push_from_locals() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(2)],
            ..Default::default()
        };

        let eval_result = super::execute_inst_push_from_local(frame, Stack::new(), 0)?;

        assert_eq!(1, eval_result.next_stack.len());
        assert_eq!(Some(Value::I32(2)), eval_result.next_stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_add_i32_illegal() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(5), Value::I32(10)],
            ..Default::default()
        };

        let eval_result = super::execute_inst_add(frame.clone(), Stack::new());

        assert_eq!(false, eval_result.is_ok());
        Ok(())
    }

    #[test]
    fn eval_inst_add_i32() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(5), Value::I32(10)],
            ..Default::default()
        };

        let eval_result = super::execute_inst_push_from_local(frame, Stack::new(), 0)?;
        let frame = eval_result.next_frame.unwrap();

        let eval_result = super::execute_inst_push_from_local(frame, eval_result.next_stack, 1)?;
        let frame = eval_result.next_frame.unwrap();

        let eval_result = super::execute_inst_add(frame, eval_result.next_stack)?;

        assert_eq!(1, eval_result.next_stack.len());
        assert_eq!(Some(Value::I32(15)), eval_result.next_stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_i32_sub() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![],
            ..Default::default()
        };

        let stack = Stack::from([Value::I32(10), Value::I32(5)]);
        let eval_result = super::execute_inst_sub(frame, stack)?;

        assert_eq!(1, eval_result.next_stack.len());
        assert_eq!(Some(Value::I32(-5)), eval_result.next_stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_i32_lt_s() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![],
            ..Default::default()
        };

        let stack = Stack::from([Value::I32(10), Value::I32(5)]);
        let eval_result = super::execute_inst_lt(frame, stack)?;

        assert_eq!(1, eval_result.next_stack.len());
        assert_eq!(Some(Value::I32(1)), eval_result.next_stack.front().map(|v| v.clone()));        
        Ok(())
    }

    #[test]
    fn make_frame_of_without_return() -> Result<()> {
        let wasm = wat::parse_str("(module (func))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        let Some(FuncInst::Internal(fn_decl)) = store.fns.get(0) else {
            unreachable!();
        };

        let expect = Frame { 
            insts: vec![
                Instruction::End,
            ], 
            ..Default::default() 
        };

        let (frame, next_stack) = super::make_frame(Stack::new(), fn_decl);
        assert_eq!(expect, frame);
        assert_eq!(0, next_stack.len());

        Ok(())
    }

    #[test]
    fn make_frame_of_with_return() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32) (result i32) (local.get 0)))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        let stack = Stack::from([Value::I32(42)]);

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

        let (frame, next_stack) = super::make_frame(stack, fn_decl);
        assert_eq!(expect, frame);
        assert_eq!(0, next_stack.len());

        Ok(())
    }

    #[test]
    fn make_frame_fn_add() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        let stack = Stack::from([Value::I32(10), Value::I32(5)]);

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

        let (frame, next_stack) = super::make_frame(stack, fn_decl);
        assert_eq!(expect, frame);
        assert_eq!(0, next_stack.len());

        Ok(())
    }

    #[test]
    fn eval_inst_local_set() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![Value::I32(0), ],
            ..Default::default()
        };
        let stack = Stack::from([Value::I32(42)]);

        let eval_result = super::execute_inst_pop_to_local(frame, stack, 0)?;

        let next_frame = eval_result.next_frame.unwrap();

        assert_eq!(0, eval_result.next_stack.len());
        assert_eq!(Value::I32(42), next_frame.locals[0]);
        Ok(())
    }

    #[test]
    fn eval_inst_i32_const() -> Result<()> {
        let frame = Frame { 
            pc: 0, 
            locals: vec![],
            ..Default::default()
        };

        let eval_result = super::execute_inst_i32_const(frame, Stack::new(), -123)?;

        assert_eq!(1, eval_result.next_stack.len());
        assert_eq!(Some(Value::I32(-123)), eval_result.next_stack.front().map(|v| v.clone()));
        Ok(())
    }

    #[test]
    fn eval_inst_i32_store() -> Result<()> {
        let stack = Stack::from([Value::I32(42), Value::I32(0)]);
        let mut memories = vec![MemoryInst { data: vec![0; 100], limit: None }];

        let frame = Frame::default();

        super::execute_inst_i32_store(frame, stack, &mut memories, 10)?;

        assert_eq!(42, memories[0].data[10]);
        Ok(())
    }

    #[test]
    fn eval_inst_end_from_fn_noreturn() -> Result<()> {
        let stack = Stack::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        // Labelが積まれていない状態でEND命令を評価する場合、関数からの脱出とみなす
        let frame = Frame { 
            labels: LinkedList::default(),
            arity: 0, sp: 0, 
            ..Default::default()
        };

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_end(frame, stack)?;
                
        assert_eq!(0, next_stack.len());

        assert_eq!(None, next_frame);

        Ok(())
    }

    #[test]
    fn eval_inst_end_from_block() -> Result<()> {
        let stack = Stack::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let frame = Frame { 
            labels: LinkedList::from([
                Label { pc: 13, arity: 1, sp: 1 }, 
                Label::default(), 
            ]), 
            arity: 0, sp: 0, 
            ..Default::default()
        };

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_end(frame, stack)?;
        
        let next_frame = next_frame.unwrap();
        
        assert_eq!(2, next_stack.len());
        assert_eq!(vec![Value::I32(42), Value::I32(44)], next_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(Frame { labels: LinkedList::from([Label::default()]), pc: 13, arity: 0, sp: 0, ..Default::default() }, next_frame);

        Ok(())
    }

    #[test]
    fn eval_inst_end_from_block_noreturn() -> Result<()> {
        let stack = Stack::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let frame = Frame { 
            labels: LinkedList::from([
                Label { pc: 99, arity: 0, sp: 1, }, 
                Label::default(), 
            ]), 
            arity: 1, sp: 0, 
            ..Default::default() 
        };

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_end(frame, stack)?;
        
        let next_frame = next_frame.unwrap();
        
        assert_eq!(1, next_stack.len());
        assert_eq!(vec![Value::I32(44)], next_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(Frame { labels: LinkedList::from([Label::default()]), pc: 99, arity: 1, sp: 0, ..Default::default() }, next_frame);
        Ok(())
    }

    #[test]
    fn eval_inst_return_from_fn() -> Result<()> {
        let stack = Stack::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let frame = Frame { 
            arity: 1, sp: 0, 
            ..Default::default() 
        };

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_return(frame, stack)?;
                
        assert_eq!(1, next_stack.len());
        assert_eq!(vec![Value::I32(42)], next_stack.into_iter().collect::<Vec<_>>());

        assert_eq!(None, next_frame);
        Ok(())
    }

    #[test]
    fn eval_inst_return_void_from_fn() -> Result<()> {
        let stack = Stack::from([Value::I32(42), Value::I32(100), Value::I64(1024), Value::I32(44)]);
        let frame = Frame { 
            arity: 0, sp: 0, 
            ..Default::default() 
        };

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_return(frame, stack)?;
                
        assert_eq!(0, next_stack.len());
        assert_eq!(None, next_frame);
        Ok(())
    }

    #[test]
    fn eval_inst_if_true() -> Result<()> {
        let stack = Stack::from([Value::I32(1)]);
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

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_if(frame, stack, &BlockType::Void)?;
        
        let next_frame = next_frame.unwrap();

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
        let stack = Stack::from([Value::I32(0)]);
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

        let EvalResultContext {next_stack, next_frame } = super::execute_inst_if(frame, stack, &BlockType::Void)?;
        
        let next_frame = next_frame.unwrap();

        assert_eq!(0, next_stack.len());

        assert_eq!(7, next_frame.pc);
        assert_eq!(0, next_frame.sp);
        assert_eq!(0, next_frame.arity);
        assert_eq!(0, next_frame.labels.len());
        Ok(())
    }
}