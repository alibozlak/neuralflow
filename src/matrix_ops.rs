use safe_matmul::matrix::Matrix;

/// Element-wise helpers safe_matmul does not have (it only multiplies),
/// built on its public `as_slice` / `from_vec`.
pub(crate) trait MatrixExt {
    /// Applies `f` to every element.
    fn map(&self, f: impl Fn(f64) -> f64) -> Matrix;

    /// Combines two matrices of the same shape element by element.
    fn zip_map(&self, other: &Matrix, f: impl Fn(f64, f64) -> f64) -> Matrix;

    /// Adds a (1 x n) row to every row: how b reaches every sample.
    fn add_row(&self, row: &Matrix) -> Matrix;

    /// (m x n) -> (1 x n), the sum of every column: how db adds up the samples.
    fn column_sums(&self) -> Matrix;

    /// A new matrix made of the given rows, in the given order.
    fn select_rows(&self, rows: &[usize]) -> Matrix;
}

impl MatrixExt for Matrix {
    fn map(&self, f: impl Fn(f64) -> f64) -> Matrix {
        let datas = self.as_slice().iter().map(|&value| f(value)).collect();
        Matrix::from_vec(self.row_count(), self.col_count(), datas).unwrap()
    }

    fn zip_map(&self, other: &Matrix, f: impl Fn(f64, f64) -> f64) -> Matrix {
        if self.shape() != other.shape() {
            panic!("Element-wise operation needs equal shapes, got {:?} and {:?} !!",
                   self.shape(), other.shape()
            );
        }

        let datas = self.as_slice().iter()
            .zip(other.as_slice())
            .map(|(&left, &right)| f(left, right))
            .collect();
        Matrix::from_vec(self.row_count(), self.col_count(), datas).unwrap()
    }

    fn add_row(&self, row: &Matrix) -> Matrix {
        if row.row_count() != 1 || row.col_count() != self.col_count() {
            panic!("Can't add a {}x{} row to every row of a {}x{} matrix !!",
                   row.row_count(), row.col_count(), self.row_count(), self.col_count()
            );
        }

        // Row-major storage: element i sits in column i % col_count.
        let datas = self.as_slice().iter()
            .zip(row.as_slice().iter().cycle())
            .map(|(&value, &addend)| value + addend)
            .collect();
        Matrix::from_vec(self.row_count(), self.col_count(), datas).unwrap()
    }

    fn column_sums(&self) -> Matrix {
        Matrix::ones(1, self.row_count()).matmul(self).unwrap()
    }

    fn select_rows(&self, rows: &[usize]) -> Matrix {
        let col_count = self.col_count();
        let mut datas = Vec::with_capacity(rows.len() * col_count);
        for &row in rows {
            datas.extend_from_slice(&self.as_slice()[row * col_count..(row + 1) * col_count]);
        }
        Matrix::from_vec(rows.len(), col_count, datas).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(row_count: usize, col_count: usize, datas: &[f64]) -> Matrix {
        Matrix::from_vec(row_count, col_count, datas.to_vec()).unwrap()
    }

    #[test]
    fn add_row_reaches_every_row() {
        let z = matrix(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let b = matrix(1, 3, &[10., 20., 30.]);
        assert_eq!(z.add_row(&b).as_slice(), &[11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    #[should_panic(expected = "Can't add a 1x2 row")]
    fn add_row_rejects_a_wrong_width() {
        matrix(2, 3, &[0.; 6]).add_row(&matrix(1, 2, &[0.; 2]));
    }

    #[test]
    fn column_sums_add_up_the_rows() {
        let dz = matrix(3, 2, &[1., 2., 3., 4., 5., 6.]);
        let db = dz.column_sums();
        assert_eq!(db.shape(), (1, 2));
        assert_eq!(db.as_slice(), &[9., 12.]);
    }

    #[test]
    fn select_rows_keeps_the_given_order() {
        let x = matrix(3, 2, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(x.select_rows(&[2, 0]).as_slice(), &[5., 6., 1., 2.]);
    }

    #[test]
    #[should_panic(expected = "equal shapes")]
    fn zip_map_rejects_different_shapes() {
        matrix(2, 3, &[0.; 6]).zip_map(&matrix(3, 2, &[0.; 6]), |a, b| a + b);
    }
}
