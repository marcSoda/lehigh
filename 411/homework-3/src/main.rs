use rand::Rng;
use homework_3::*;

fn main() {
    // matrix sizes to be benched
    let ns = vec!(10, 25, 50, 100, 250, 500, 1000, 2000, 4000);
    for n in ns {
        println!("BENCHING {}x{} MATRIX", n, n);
        // Generate a random matrix and vector of size n
        let a: Vec<Vec<f64>> = (0..n).map(|_| {
            (0..n).map(|_| rand::thread_rng().gen_range(-10.0..10.0)).collect()
        }).collect();
        let b: Vec<f64> = (0..n).map(|_| rand::thread_rng().gen_range(-10.0..10.0)).collect();
        // Run benches
        print!("\trunning seq... ");
        let seq_start = std::time::Instant::now();
        let _seq_sol = gauss(a.clone(), b.clone(), vec!());
        let seq_time = seq_start.elapsed().as_micros();
        print!("took {:?}µs\n", seq_time);
        print!("\trunning fp...  ");
        let fp_start = std::time::Instant::now();
        let _fp_sol = gauss(a.clone(), b.clone(), vec!(Par::FindPiv));
        let fp_time = fp_start.elapsed().as_micros();
        print!("took {:?}µs\n", fp_time);
        print!("\trunning npr... ");
        let npr_start = std::time::Instant::now();
        let _npr_sol = gauss(a.clone(), b.clone(), vec!(Par::NormPivRow));
        let npr_time = npr_start.elapsed().as_micros();
        print!("took {:?}µs\n", npr_time);
        print!("\trunning ebp... ");
        let ebp_start = std::time::Instant::now();
        let _ebp_sol = gauss(a.clone(), b.clone(), vec!(Par::ElimBelowPiv));
        let ebp_time = ebp_start.elapsed().as_micros();
        print!("took {:?}µs\n", ebp_time);
        print!("\trunning bs...  ");
        let bs_start = std::time::Instant::now();
        let _bs_sol = gauss(a.clone(), b.clone(), vec!(Par::BackSub));
        let bs_time = bs_start.elapsed().as_micros();
        print!("took {:?}µs\n", bs_time);
        print!("\trunning all... ");
        let all_start = std::time::Instant::now();
        let _all_sol = gauss(a.clone(), b.clone(), vec!(Par::FindPiv, Par::NormPivRow, Par::ElimBelowPiv, Par::BackSub));
        let all_time = all_start.elapsed().as_micros();
        print!("took {:?}µs\n", all_time);
    }
}
