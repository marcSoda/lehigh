use std::cmp::max;
use std::cmp::min;
use std::fs;
use std::path::Path;
use std::str::FromStr;

type Error = Box<dyn std::error::Error>;

pub type Result<T> = std::result::Result<T, Error>;

pub struct Matrix {
    labels: Vec<String>,
    dists: Vec<Vec<f32>>,
}

impl Matrix {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn new(labels: Vec<String>, dists: Vec<Vec<f32>>) -> Self {
        Self { labels, dists }
    }

    pub fn from_file(p: &Path) -> Result<Self> {
        fs::read_to_string(p)?.parse()
    }
}

//read in lower triangular matrix as full symmetric square matrix
impl FromStr for Matrix {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let mut lines = s.lines();
        let n = lines.next().ok_or("Expected n")?.replace(" ", "").parse()?;
        let mut d = Matrix::new(Vec::with_capacity(n), vec![vec![0.0; n]; n]);

        for (i, line) in lines.into_iter().enumerate() {
            let label = line
                .chars()
                .take(10)
                .filter(|c| !c.is_whitespace())
                .collect::<String>();
            d.labels.push(label);
            let row: Vec<f32> = line[10..]
                .split_whitespace()
                .map(|s| s.parse::<f32>().unwrap())
                .collect();
            for (j, &dist) in row.iter().enumerate() {
                d.dists[i][j] = dist;
                d.dists[j][i] = dist;  // mirror
            }
        }
        Ok(d)
    }
}

#[derive(PartialEq, Debug)]
pub struct Newick {
    label: String,
    kids: Vec<(Newick, Option<f32>)>,
}

impl Newick {
    pub fn new(label: &str, children: Vec<(Newick, Option<f32>)>) -> Newick {
        Newick { label: label.to_string(), kids: children }
    }

    pub fn new_leaf(label: &str) -> Newick {
        Newick::new(label, vec![])
    }

    pub fn join_label(label: &str, l: Newick, dl: f32, r: Newick, dr: f32) -> Newick {
        Newick::new(label, vec![(l, Some(dl)), (r, Some(dr))])
    }

fn str(&self) -> String {
    if self.kids.is_empty() {
        return self.label.to_string();
    }

    let children: Vec<String> = self.kids.iter().map(|(child, dist)| {
        let child_str = Self::str(child);
        if let Some(dist) = dist {
            format!("{}:{}", child_str, dist)
        } else {
            child_str
        }
    }).collect();

    format!("({}){}", children.join(","), self.label)
}
}

impl ToString for Newick {
    fn to_string(&self) -> String {
        Newick::str(&self) + ";"
    }
}

pub fn neighbor(dists: &Matrix) -> Newick {
    let mut parts: Vec<Option<Newick>> = dists
        .labels
        .iter()
        .map(|l| Some(Newick::new_leaf(l)))
        .collect();
    let mut dists = dists.dists.clone();

    let mut active_indices: Vec<usize> = (0..dists.len()).collect();

    while active_indices.len() > 2 {
        let compute_sum = |index: usize| -> f32 { active_indices.iter().map(|&k| dists[index][k]).sum::<f32>() };

        let q_func = |i: &usize, j: &usize| -> f32 {
            (active_indices.len() - 2) as f32 * dists[*i][*j] - compute_sum(*i) - compute_sum(*j)
        };

        let mut min_value = std::f32::MAX;
        let mut min_indices = (0, 0);
        for i in &active_indices {
            for j in &active_indices {
                if i != j {
                    let current_q = q_func(i, j);
                    if current_q < min_value {
                        min_value = current_q;
                        min_indices = (*i, *j);
                    }
                }
            }
        }

        let (idx_i, idx_j) = (min(min_indices.0, min_indices.1), max(min_indices.0, min_indices.1));

        let di = dists[idx_i][idx_j] / 2. + (compute_sum(idx_i) - compute_sum(idx_j)) / (2. * (active_indices.len() as f32 - 2.));
        let dj = dists[idx_i][idx_j] - di;

        active_indices.remove(active_indices.iter().position(|&x| x == idx_j).unwrap());

        active_indices.iter().filter(|&&k| k != idx_i).for_each(|&k| {
            let dk = (dists[idx_i][k] + dists[idx_j][k] - dists[idx_i][idx_j]) / 2.;
            dists[idx_i][k] = dk;
            dists[k][idx_i] = dk;
        });
        parts[idx_i] = Some(Newick::join_label(
            "",
            parts[idx_i].take().unwrap(),
            di,
            parts[idx_j].take().unwrap(),
            dj,
        ));
    }

    if let [idx_i, idx_j] = active_indices[..] {
        let d = dists[idx_i][idx_j] / 2.;
        parts[idx_i] = Some(Newick::join_label(
            "",
            parts[idx_i].take().unwrap(),
            d,
            parts[idx_j].take().unwrap(),
            d,
        ));
    }

    parts[0].take().unwrap()
}

pub fn upgma(distances: &Matrix) -> Newick {
    let n = distances.len();
    let mut parts: Vec<Option<(Newick, usize, f32)>> = distances
        .labels
        .iter()
        .map(|name| Some((Newick::new_leaf(name), 1, 0.0)))
        .collect();
    let mut dists = distances.dists.clone();
    let mut up = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dists[i][j];
            if !d.is_nan() {
                up.push((d, i, j));
            }
        }
    }
    up.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    while let Some((d, i, j)) = up.pop() {
        if parts[i].is_none() || parts[j].is_none() || dists[i][j] != d {
            continue;
        }
        let (i, j) = (i.min(j), i.max(j));
        if let (Some((pi, si, di)), Some((pj, sj, dj))) = (parts[i].take(), parts[j].take()) {
            let new_label = Newick::join_label("", pi, d / 2. - di, pj, d / 2. - dj);
            let new_size = si + sj;
            let new_distance = d / 2.;

            parts[i] = Some((new_label, new_size, new_distance));

            for k in 0..n {
                if k == i || k == j {
                    continue;
                }
                let dk = (si as f32 * dists[i][k] + sj as f32 * dists[j][k]) / (si + sj) as f32;
                dists[i][k] = dk;
                dists[k][i] = dk;
                up.push((dk, k, i));
            }
            up.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }
    }

    parts[0].take().unwrap().0
}
