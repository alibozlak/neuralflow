use std::fmt;
use std::str::FromStr;
use safe_matmul::matrix::Matrix;
use crate::matrix_ops::MatrixExt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Sigmoid,
    Linear,
    ReLU,
}

impl Activation {

    /// a = g(z)
    pub fn a(self, z: f64) -> f64 {

        match self {
            Activation::Sigmoid => 1. / (1. + (-z).exp()),
            Activation::Linear => z,
            Activation::ReLU => z.max(0.0),
        }
    }

    /// A = g(Z), element by element.
    pub fn a_with_matrix(self, z_matrix: &Matrix) -> Matrix {
        z_matrix.map(|z| self.a(z))
    }

    /// g'(z): backprop multiplies dA by this to get dZ.
    pub fn derivative(self, z: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let a = self.a(z);
                a * (1. - a)
            },
            Activation::Linear => 1.,
            // ReLU has no slope at exactly 0; TensorFlow uses 0 there too.
            Activation::ReLU => if z > 0. { 1. } else { 0. },
        }
    }

    /// g'(Z), element by element.
    pub fn derivative_with_matrix(self, z_matrix: &Matrix) -> Matrix {
        z_matrix.map(|z| self.derivative(z))
    }

    /// dZ = dA * g'(Z): backprop's step back through the activation.
    pub fn dz_matrix(self, da_matrix: &Matrix, z_matrix: &Matrix) -> Matrix {
        da_matrix.zip_map(&self.derivative_with_matrix(z_matrix), |da, slope| da * slope)
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sigmoid => "Sigmoid",
            Self::Linear => "Linear",
            Self::ReLU => "ReLU",
        };
        write!(f, "{name}")
    }
}

impl FromStr for Activation {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "Sigmoid" => Ok(Self::Sigmoid),
            "Linear" => Ok(Self::Linear),
            "ReLU" => Ok(Self::ReLU),
            other => Err(format!("unknown activation: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matches_the_formulas() {
        assert_eq!(Activation::Sigmoid.a(0.), 0.5);
        assert_eq!(Activation::Linear.a(-3.), -3.);
        assert_eq!(Activation::ReLU.a(-3.), 0.);
        assert_eq!(Activation::ReLU.a(3.), 3.);
    }

    /// g'(z) against the slope (g(z + h) - g(z - h)) / 2h.
    #[test]
    fn derivative_matches_the_numerical_slope() {
        let h = 1e-6;
        for activation in [Activation::Sigmoid, Activation::Linear, Activation::ReLU] {
            // ReLU has a kink at 0, so stay away from it.
            for z in [-2.5, -0.3, 0.7, 4.] {
                let slope = (activation.a(z + h) - activation.a(z - h)) / (2. * h);
                assert!((activation.derivative(z) - slope).abs() < 1e-6, "{activation} at z = {z}");
            }
        }
    }

    #[test]
    fn matrix_versions_work_element_by_element() {
        let z = Matrix::from_vec(2, 2, vec![-1., 0., 1., 2.]).unwrap();
        assert_eq!(Activation::ReLU.a_with_matrix(&z).as_slice(), &[0., 0., 1., 2.]);
        assert_eq!(Activation::ReLU.derivative_with_matrix(&z).as_slice(), &[0., 0., 1., 1.]);
    }
}
