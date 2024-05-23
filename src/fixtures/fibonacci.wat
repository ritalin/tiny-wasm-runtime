(module
    (func $fib (export "fib") (param $n i32) (result i32)
        (i32.lt_s (local.get $n) (i32.const 2))
        (if 
            (then (return (i32.const 1)))
        )
        (return
            (i32.add
                (call $fib (i32.sub (local.get $n) (i32.const 2)))
                (call $fib (i32.sub (local.get $n) (i32.const 1)))
            )
        )
    )
    (export "_start" (func $fib))
)
