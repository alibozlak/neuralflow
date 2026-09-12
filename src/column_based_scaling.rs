use safe_matmul::matrix::Matrix;

pub fn manipulate_datas_between_0_and_10(input_matrix : Matrix, output_matrix : Matrix)
                                         -> (Matrix, Matrix, Vec<usize>)
{
    validate_input_matrix_and_output_matrix(&input_matrix, &output_matrix);

    let m = input_matrix.row_count();
    let n = input_matrix.col_count();
    let mut result_inputs : Vec<Vec<f64>> = vec![vec![0.0; n]; m];
    let mut result_outputs : Vec<f64> = vec![0.0; m];
    let mut ten_power_ratios : Vec<usize> = vec![0; n + 1];

    for i in 0..n {
        let (scaled_data, ratio) = find_column_ratio(input_matrix.get(0,i).unwrap());
        result_inputs[0][i] = scaled_data;
        ten_power_ratios[i] = ratio;
    }
    let (scaled_data, ratio) = find_column_ratio(output_matrix.get(0,0).unwrap());
    result_outputs[0] = scaled_data;
    ten_power_ratios[n] = ratio;

    for i in 1..m {
        for j in 0..n {
            result_inputs[i][j] = convert_data(input_matrix.get(i,j).unwrap(), ten_power_ratios[j]);
        }

        result_outputs[i] = convert_data(output_matrix.get(i,0).unwrap(), ten_power_ratios[n]);
    }

    let mut result_input_vec: Vec<f64> = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            result_input_vec.push(result_inputs[i][j]);
        }
    }

    (
        Matrix::from_vec(m, n, result_input_vec).unwrap(),
        Matrix::from_vec(m, 1, result_outputs).unwrap(),
        ten_power_ratios
    )
}

fn convert_data(data : f64, ten_power_ratio : usize) -> f64 {
    data * 10.0_f64.powi(-(ten_power_ratio as i32))
}

fn find_column_ratio(data : f64) -> (f64, usize) {
    let mut ratio : usize = 0;
    let positive_data = data.abs();
    let mut data_s_string = positive_data.to_string();
    if let Some(index) = data_s_string.find(".") {
        data_s_string = data_s_string.split_at(index).0.to_string();
    }
    if data_s_string.len() > 1 { ratio = data_s_string.len() - 1; }

    (convert_data(data, ratio), ratio)
}

fn validate_input_matrix_and_output_matrix(input_matrix : &Matrix, output_matrix : &Matrix) {
    if input_matrix.row_count() != output_matrix.row_count() {
        panic!("The number of rows in the input matrix and the output matrix must be the same !!");
    }

    if output_matrix.col_count() != 1 {
        panic!("The number of columns in the output matrix must be 1 !!");
    }
}