use h1::Set;
use rand::Rng;
use std::time::Duration;
use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    Criterion
};
use h1::{
    array_set::ArraySet,
    hash_set::HashSet,
    list_set::ListSet,
    tree_set::TreeSet,
    SetType,
};

fn run_operations(set: &mut dyn Set<usize>, i: usize, k: usize, r: usize) {
    let mut rng = rand::thread_rng();
    for _ in 0..i {
        let val: usize = rng.gen_range(0..k);
        let op_type: usize = rng.gen_range(0..100);

        if op_type < r {
            set.find(black_box(val));
        } else if op_type < (100 + r) / 2 {
            set.insert(black_box(val));
        } else {
            set.remove(black_box(val));
        }
    }
}

fn populate_set(set: &mut dyn Set<usize>, k: usize, s: usize) {
    let mut rng = rand::thread_rng();
    for _ in 0..(s / 2) {
        let val: usize = rng.gen_range(0..k);
        set.insert(val);
    }
}

fn bench(c: &mut Criterion, typ: SetType, test_prefix: String, i: usize, slist: Vec<usize>, rlist: Vec<usize>, k: usize) {
    for s in 0..slist.len() {
        for r in 0..rlist.len() {
            let mut set: Box<dyn Set<usize>> = match typ {
                SetType::ArraySet => Box::new(ArraySet::new()),
                SetType::HashSet => Box::new(HashSet::new()),
                SetType::ListSet => Box::new(ListSet::new()),
                SetType::TreeSet => Box::new(TreeSet::new()),
            };
            populate_set(&mut *set, k, slist[s]);
            c.bench_function(&(test_prefix.clone() + &slist[s].to_string() + "_" + &rlist[r].to_string()),
                             |b| b.iter(|| run_operations(black_box(&mut *set), i, k, rlist[r])));
        }
    }
}

fn bench_all(c: &mut Criterion) {
    const K: usize = 10000;
    const I: usize = 10000; // num ops
    let slist: Vec<usize> = vec![1000, 10000, 100000, 1000000];
    let rlist: Vec<usize> = vec![0, 20, 50, 80, 100];

    bench(c, SetType::ArraySet, "array_set_".to_string(), I, slist.clone(), rlist.clone(), K);
    bench(c, SetType::HashSet, "hash_set_".to_string(), I, slist.clone(), rlist.clone(), K);
    bench(c, SetType::ListSet, "list_set_".to_string(), I, slist.clone(), rlist.clone(), K);
    bench(c, SetType::TreeSet, "tree_set_".to_string(), I, slist.clone(), rlist.clone(), K);
}

criterion_group!{
  name = benches;
  config = Criterion::default().measurement_time(Duration::from_secs(1))
                               .sample_size(10)
                               .warm_up_time(Duration::from_secs(1));
  targets = bench_all
}
criterion_main!(benches);
