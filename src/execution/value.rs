
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

impl From<Value> for i32 {
    fn from(value: Value) -> Self {
        match value {
            Value::I32(v) => v,
            _ => unimplemented!("Unsupported feature"),
        }
    }
}
