use rayon::prelude::*;

#[derive(PartialEq)]
pub enum Par {
    FindPiv,
    NormPivRow,
    ElimBelowPiv,
    BackSub
}

// whichever `Par` is in `pars`, the corresponding parallelized method will be called rather than the sequential variant
pub fn gauss(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, pars: Vec<Par>) -> Result<Vec<f64>, String> {
    let n = b.len();

    // Forward Elimination (Eliminate variables below the diagonal)
    for pivot_row in 0..n {
        // Find the pivot element
        let pivot_index = match pars.contains(&Par::FindPiv) {
            true => par_find_pivot(&a, pivot_row, n),
            false => find_pivot(&a, pivot_row, n),
        };

        // Swap rows if necessary
        if pivot_index != pivot_row {
            a.swap(pivot_row, pivot_index);
            b.swap(pivot_row, pivot_index);
        }

        let pivot_value = a[pivot_row][pivot_row];
        if pivot_value == 0.0 {
            return Err("Matrix is singular or poorly conditioned.".to_string());
        }

        // Normalize the pivot row
        match pars.contains(&Par::NormPivRow) {
            true => par_normalize_pivot_row(&mut a, &mut b, pivot_row, pivot_value, n),
            false => normalize_pivot_row(&mut a, &mut b, pivot_row, pivot_value, n),
        };

        // Eliminate variables below the pivot row
        match pars.contains(&Par::ElimBelowPiv) {
            true => par_eliminate_below_pivot(&mut a, &mut b, pivot_row, n),
            false => eliminate_below_pivot(&mut a, &mut b, pivot_row, n),
        };
    }

    // Backward Substitution (Solve for variables)
    let x = match pars.contains(&Par::BackSub) {
        true => par_backward_substitution(&a, &b, n),
        false => backward_substitution(&a, &b, n),
    };
    Ok(x)
}

fn find_pivot(a: &Vec<Vec<f64>>, pivot_row: usize, n: usize) -> usize {
    let mut pivot_index = pivot_row;
    for i in (pivot_row + 1)..n {
        if a[i][pivot_row].abs() > a[pivot_index][pivot_row].abs() {
            pivot_index = i;
        }
    }
    pivot_index
}

fn normalize_pivot_row(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, pivot_row: usize, pivot_value: f64, n: usize) {
    for j in pivot_row..n {
        a[pivot_row][j] /= pivot_value;
    }
    b[pivot_row] /= pivot_value;
}

fn eliminate_below_pivot(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, pivot_row: usize, n: usize) {
    for i in (pivot_row + 1)..n {
        let factor = a[i][pivot_row];
        for j in pivot_row..n {
            a[i][j] -= factor * a[pivot_row][j];
        }
        b[i] -= factor * b[pivot_row];
    }
}

fn backward_substitution(a: &Vec<Vec<f64>>, b: &Vec<f64>, n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
    }
    x
}

fn par_find_pivot(a: &Vec<Vec<f64>>, pivot_row: usize, n: usize) -> usize {
    (pivot_row + 1..n).into_par_iter()
        .max_by(|&x, &y| a[x][pivot_row].abs().partial_cmp(&a[y][pivot_row].abs()).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(pivot_row)
}

fn par_normalize_pivot_row(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, pivot_row: usize, pivot_value: f64, n: usize) {
    a[pivot_row][pivot_row..n].par_iter_mut().for_each(|value| *value /= pivot_value);
    b[pivot_row] /= pivot_value;
}

fn par_eliminate_below_pivot(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, pivot_row: usize, n: usize) {
    let pivot_a_row = a[pivot_row].clone();
    let pivot_b_val = b[pivot_row];
    a.par_iter_mut().skip(pivot_row + 1).zip(b.par_iter_mut().skip(pivot_row + 1)).for_each(|(a_row, b_val)| {
        let factor = a_row[pivot_row];
        for j in pivot_row..n { a_row[j] -= factor * pivot_a_row[j]; }
        *b_val -= factor * pivot_b_val;
    });
}

fn par_backward_substitution(a: &Vec<Vec<f64>>, b: &Vec<f64>, n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let total_sum: f64 = (i+1..n)
            .into_par_iter()
            .map(|j| a[i][j] * x[j])
            .reduce(|| 0.0, |sum, val| sum + val);
        x[i] = b[i] - total_sum;
    }
    x
}
