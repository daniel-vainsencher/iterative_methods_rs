use iterative_methods::utils::expose_w;
use iterative_methods::*;
use streaming_iterator::*;
use utils::*;

fn wd_iterable_counter_demo() {
    println!("\n\n -----Weight Counter Demo----- \n\n");

    let counter_stream = Counter::new();
    let mut counter_stream_copy = counter_stream;
    let mut wd_iter = Weight {
        it: counter_stream,
        f: expose_w,
        wd: Some(new_datum(0., 0.)),
    };

    println!("A stream of values:\n");
    for _ in 0..6 {
        if let Some(val) = counter_stream_copy.next() {
            println!("{}", val);
        }
    }
    println!("\n\nThe stream with weights added that are the square of the value.\n\n");
    for _ in 0..6 {
        if let Some(wd) = wd_iter.next() {
            println!("{:#?}", wd);
        }
    }
}

// In Process: a demo using a stream of strings; weight is length
// fn wd_iterable_string_demo() {
//
// }

fn main() {
    wd_iterable_counter_demo();
}
