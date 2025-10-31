#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <armadillo>
#include <omp.h>

using namespace arma;
using namespace std;

// configuration
struct Config {
    string input_file;
    string output_file;
    int n_pcs;
    string covs_file;
    int n_hvg = 2000;
    int n_bins = 20;
    double target_sum = 1e4;
    double max_scale = 10.0;
};

// HVG results
struct HVGResult {
    uvec hvg_indices;
    vec means;
    vec dispersions;
    vec dispersions_norm;
};

// TSV reading
mat read_tsv_optimized(const string& filename, vector<string>& row_names, 
                      vector<string>& col_names) {
    
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    // File in memory - this we may want to change and process in chunks
    file.seekg(0, ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    
    string content(file_size, ' ');
    file.read(&content[0], file_size);
    file.close();
    
    // Parse
    istringstream stream(content);
    string line;
    
    // Read header
    getline(stream, line);
    istringstream header_stream(line);
    string token;
    getline(header_stream, token, '\t'); // Skip row names header
    while (getline(header_stream, token, '\t')) {
        col_names.push_back(token);
    }
    
    // Count rows first for pre-allocation
    size_t n_rows = 0;
    size_t pos = stream.tellg();
    while (getline(stream, line)) n_rows++;
    stream.clear();
    stream.seekg(pos);
    
    size_t n_cols = col_names.size();
    
    // Pre-allocate matrix and vectors
    mat data(n_rows, n_cols);
    row_names.reserve(n_rows);
    
    // Reading
    size_t row_idx = 0;
    while (getline(stream, line)) {
        istringstream line_stream(line);
        
        getline(line_stream, token, '\t');
        row_names.push_back(token);
        
        size_t col_idx = 0;
        while (getline(line_stream, token, '\t') && col_idx < n_cols) {
            data(row_idx, col_idx) = stod(token);
            col_idx++;
        }
        row_idx++;
    }
    
    return data;
}

// Read existing covariates
vector<string> read_covariates(const string& covs_file) {
    ifstream file(covs_file);
    string line;
    getline(file, line);
    file.close();
    
    vector<string> covs;
    istringstream iss(line);
    string token;
    while (getline(iss, token, ',')) {
        covs.push_back(token);
    }
    return covs;
}

// Normalization with vectorized operations
void normalize_total_optimized(mat& data, double target_sum) {
    vec row_sums = sum(data, 1);
    
    // Vectorized normalization
    for (uword i = 0; i < data.n_rows; ++i) {
        if (row_sums(i) > 0) {
            data.row(i) *= (target_sum / row_sums(i));
        }
    }
}

// Armadillo's built-in vectorized log1p
void log1p_transform_optimized(mat& data) {
    data = log1p(data);
}

// HVG selection
HVGResult select_hvg_seurat_optimized(const mat& data, int n_top_genes, int n_bins = 20) {
    uword n_genes = data.n_cols;
    
    // Use Armadillo's built-in statistics
    rowvec means = mean(data, 0);
    rowvec vars = var(data, 0, 0);
    
    vec means_vec = means.t();
    vec vars_vec = vars.t();
    vec dispersions = vars_vec / (means_vec + 1e-12);
    
    // Binning
    vec mean_sorted = sort(means_vec);
    vec bin_edges(n_bins + 1);
    
    for (int i = 0; i <= n_bins; ++i) {
        double quantile = (double)i / n_bins;
        uword idx = (uword)(quantile * (mean_sorted.n_elem - 1));
        bin_edges(i) = mean_sorted(idx);
    }
    bin_edges(n_bins) = mean_sorted(mean_sorted.n_elem - 1) + 1e-6;
    
    vec dispersions_norm = zeros<vec>(n_genes);
    
    // Bin processing
    for (int bin = 0; bin < n_bins; ++bin) {
        uvec bin_genes = find(means_vec >= bin_edges(bin) && means_vec < bin_edges(bin + 1));
        
        if (bin_genes.n_elem > 1) {
            vec bin_dispersions = dispersions(bin_genes);
            double bin_mean = mean(bin_dispersions);
            double bin_std = stddev(bin_dispersions);
            
            if (bin_std > 1e-12) {
                dispersions_norm(bin_genes) = (bin_dispersions - bin_mean) / bin_std;
            }
        } else if (bin_genes.n_elem == 1) {
            dispersions_norm(bin_genes(0)) = 1.0;
        }
    }
    
    uvec sorted_indices = sort_index(dispersions_norm, "descend");
    uvec hvg_indices = sorted_indices.head(min((uword)n_top_genes, n_genes));
    
    HVGResult result;
    result.hvg_indices = hvg_indices;
    result.means = means_vec;
    result.dispersions = dispersions;
    result.dispersions_norm = dispersions_norm;
    
    return result;
}

// Scaling
void scale_data_optimized(mat& data, double max_value) {
    // Armadillo's built-in standardization
    rowvec col_means = mean(data, 0);
    rowvec col_stds = stddev(data, 0, 0);
    
    for (uword j = 0; j < data.n_cols; ++j) {
        if (col_stds(j) > 1e-12) {
            data.col(j) = (data.col(j) - col_means(j)) / col_stds(j);
            
            // Vectorized clipping
            data.col(j) = clamp(data.col(j), -max_value, max_value);
        }
    }
}

// PCA
mat compute_pca_optimized(const mat& data, int n_pcs, vec& eigenvalues) {
    // Center data efficiently
    mat centered = data.each_row() - mean(data, 0);
    
    // Use more efficient SVD approach for tall matrices
    mat cov_matrix, U, V;
    vec s;
    
    if (centered.n_rows > centered.n_cols) {
        // More genes than cells - compute covariance matrix first
        cov_matrix = centered.t() * centered / (centered.n_rows - 1);
        bool success = eig_sym(s, U, cov_matrix);
        if (!success) throw runtime_error("Eigendecomposition failed");
        
        // Sort eigenvalues and eigenvectors in descending order
        uvec indices = sort_index(s, "descend");
        s = s(indices);
        U = U.cols(indices);
        
        // Project data onto eigenvectors
        int n_components = min(n_pcs, (int)s.n_elem);
        mat pca_scores = centered * U.cols(0, n_components - 1);
        eigenvalues = s.head(n_components);
        
        return pca_scores;
    } else {
        // More cells than genes - use standard SVD
        bool success = svd_econ(U, s, V, centered.t(), "both", "std");
        if (!success) throw runtime_error("SVD computation failed");
        
        int n_components = min(n_pcs, (int)min(U.n_cols, V.n_rows));
        mat pca_scores = V.cols(0, n_components - 1);
        eigenvalues = square(s.head(n_components)) / (data.n_rows - 1);
        
        return pca_scores;
    }
}

// Writing
void write_output_optimized(const string& filename, const mat& original_data,
                           const mat& pca_loadings, const mat& genotype_pcs,
                           const vector<string>& row_names, 
                           const vector<string>& orig_col_names,
                           int n_pcs, const vector<string>& geno_cov_names) {
    
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open output file: " + filename);
    }
    
    // Set precision and format once
    file.precision(6);
    file << std::fixed;
    
    // Build header string once
    string header = "sample_id";
    for (const auto& name : orig_col_names) {
        header += "\t" + name;
    }
    for (int i = 0; i < n_pcs; ++i) {
        header += "\txPC" + to_string(i + 1);
    }
    for (const auto& name : geno_cov_names) {
        header += "\t" + name;
    }
    header += "\n";
    file << header;
    
    // Write data rows with minimal string operations
    for (size_t i = 0; i < original_data.n_rows; ++i) {
        file << row_names[i];
        
        // Original data
        for (uword j = 0; j < original_data.n_cols; ++j) {
            file << "\t" << original_data(i, j);
        }
        
        // PCA loadings
        for (int j = 0; j < n_pcs; ++j) {
            file << "\t" << pca_loadings(i, j);
        }
        
        // Genotype PCs
        for (uword j = 0; j < genotype_pcs.n_cols; ++j) {
            file << "\t" << genotype_pcs(i, j);
        }
        file << "\n";
    }
    
    file.close();
}

// Main
void run_pca_analysis_optimized(const Config& config) {
    cout << "Reading input data..." << endl;
    
    vector<string> row_names, col_names;
    mat counts = read_tsv_optimized(config.input_file, row_names, col_names);
    
    cout << "Data shape: " << counts.n_rows << " x " << counts.n_cols << endl;
    
    // Read and extract covariates
    vector<string> existing_covs = read_covariates(config.covs_file);
    
    // Find covariate columns efficiently
    vector<uword> cov_indices, gene_indices;
    vector<string> gene_col_names;
    
    cov_indices.reserve(existing_covs.size());
    gene_indices.reserve(col_names.size());
    gene_col_names.reserve(col_names.size());
    
    for (size_t i = 0; i < col_names.size(); ++i) {
        bool is_cov = false;
        for (const auto& cov : existing_covs) {
            if (col_names[i] == cov) {
                cov_indices.push_back(i);
                is_cov = true;
                break;
            }
        }
        if (!is_cov) {
            gene_indices.push_back(i);
            gene_col_names.push_back(col_names[i]);
        }
    }
    
    // Extract matrices
    mat genotype_pcs = counts.cols(conv_to<uvec>::from(cov_indices));
    mat counts_orig = counts.cols(conv_to<uvec>::from(gene_indices));
    
    cout << "Processing " << counts_orig.n_cols << " genes..." << endl;
    
    // Keep original data for output
    mat original_data = counts_orig;
    
    // Process pipeline
    cout << "Normalizing..." << endl;
    normalize_total_optimized(counts_orig, config.target_sum);
    
    cout << "Log transforming..." << endl;
    log1p_transform_optimized(counts_orig);
    
    cout << "Selecting highly variable genes (Seurat flavor)..." << endl;
    HVGResult hvg_result = select_hvg_seurat_optimized(counts_orig, config.n_hvg, config.n_bins);
    mat hvg_data = counts_orig.cols(hvg_result.hvg_indices);
    
    cout << "Selected " << hvg_data.n_cols << " HVGs" << endl;
    cout << "Top 5 HVG normalized dispersions: ";
    for (int i = 0; i < min(5, (int)hvg_result.hvg_indices.n_elem); ++i) {
        cout << hvg_result.dispersions_norm(hvg_result.hvg_indices(i)) << " ";
    }
    cout << endl;
    
    cout << "Scaling..." << endl;
    scale_data_optimized(hvg_data, config.max_scale);
    
    cout << "Computing PCA..." << endl;
    vec eigenvalues;
    mat pca_loadings = compute_pca_optimized(hvg_data, config.n_pcs, eigenvalues);
    
    cout << "PCA complete. Variance explained by first 5 PCs: ";
    for (int i = 0; i < min(5, (int)eigenvalues.n_elem); ++i) {
        cout << eigenvalues(i) << " ";
    }
    cout << endl;
    
    cout << "Writing output..." << endl;
    write_output_optimized(config.output_file, original_data, pca_loadings, 
                          genotype_pcs, row_names, gene_col_names, 
                          config.n_pcs, existing_covs);
    
    // Write covariates file
    ofstream cov_file("covariates_new.txt");
    string cov_line1, cov_line2;
    for (size_t i = 0; i < existing_covs.size(); ++i) {
        if (i > 0) {
            cov_line1 += ",";
            cov_line2 += ",";
        }
        cov_line1 += existing_covs[i];
        cov_line2 += existing_covs[i];
    }
    for (int i = 0; i < config.n_pcs; ++i) {
        cov_line1 += ",xPC" + to_string(i + 1);
    }
    cov_file << cov_line1 << "\n" << cov_line2 << endl;
    cov_file.close();
    
    cout << "Done!" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file> <n_pcs> <covs_file>" << endl;
        return 1;
    }
    
    // Limit OpenMP threads to reasonable number
    omp_set_num_threads(min(4, omp_get_max_threads()));
    
    Config config;
    config.input_file = argv[1];
    config.output_file = argv[2];
    config.n_pcs = stoi(argv[3]);
    config.covs_file = argv[4];
    
    try {
        run_pca_analysis_optimized(config);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}

