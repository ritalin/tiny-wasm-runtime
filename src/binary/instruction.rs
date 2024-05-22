use super::types::Block;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    End,
    If(Block),
    Return,
    Call(u32),
    LocalGet(u32),
    LocalSet(u32),
    I32Store { align: u32, offset: u32 },
    I32Const(i32),
    I32LtS,
    I32Add,
    I32Sub,
}