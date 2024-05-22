
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    I32(i32),
    I64(i64),
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::I32(lhs), Value::I32(rhs)) => Value::I32(lhs + rhs),
            (Value::I64(lhs), Value::I64(rhs)) => Value::I64(lhs + rhs),
            _ => unreachable!()
        }
    }
}

impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::I32(lhs), Value::I32(rhs)) => Value::I32(lhs - rhs),
            (Value::I64(lhs), Value::I64(rhs)) => Value::I64(lhs - rhs),
            _ => unreachable!()
        }        
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        match (self, rhs) {
            (Value::I32(lhs), Value::I32(rhs)) => lhs.partial_cmp(&rhs),
            (Value::I64(lhs), Value::I64(rhs)) => lhs.partial_cmp(&rhs),
            _ => None,
        }
    }
}

impl From<Value> for i32 {
    fn from(value: Value) -> Self {
        match value {
            Value::I32(v) => v,
            _ => unimplemented!("Unsupported feature"),
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::I32(Into::<i32>::into(value))
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Label {
    pub arity: u32,
    pub sp: usize,
}