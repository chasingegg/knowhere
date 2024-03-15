// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstdlib>
#include <fstream>
#include <string_view>
#include <vector>

#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/file_manager.h"

namespace knowhere::kmeans {
namespace {

// 10 million 700dim float vector data takes about 26G memory
// set it to the max train size for kmeans clustering
static constexpr int64_t MAX_TRAIN_SIZE = 10000000ULL * 700 * 4;

// load file to pre-allocated data buffer + offset
template <typename T>
inline bool
load_bin_file(const std::string& fname, std::unique_ptr<T[]>& data, uint64_t& offset) {
    std::ifstream reader(fname, std::ios::binary | std::ios::ate);
    size_t actual_file_size;
    if (!reader.fail() && reader.is_open()) {
        actual_file_size = reader.tellg();
    } else {
        LOG_KNOWHERE_ERROR_ << "Could not open file: " << fname;
        return false;
    }
    reader.seekg(0, std::ios::beg);
    uint64_t npts, dim;
    uint32_t npts_32, dim_32;
    // reader.read((char*)&npts, sizeof(uint64_t));
    // reader.read((char*)&npts, sizeof(uint64_t));

    reader.read((char*)&npts_32, sizeof(uint32_t));
    reader.read((char*)&dim_32, sizeof(uint32_t));
    npts = npts_32;
    dim = dim_32;

    LOG_KNOWHERE_INFO_ << "Metadata: #pts = " << npts << ", #dims = " << dim;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
        LOG_KNOWHERE_ERROR_ << "Error. File size mismatch. Actual size is " << actual_file_size
                            << " while expected size is  " << expected_actual_file_size;
        reader.close();
        return false;
    }

    reader.read(reinterpret_cast<char*>(data.get() + offset * dim), npts * dim * sizeof(T));
    offset += npts;
    reader.close();
    return true;
}

inline bool
get_bin_metadata(const std::string& bin_file, uint64_t& nrows, uint64_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    if (!reader.fail() && reader.is_open()) {
        uint32_t nrows_32, ncols_32;
        // reader.read((char*)&nrows, sizeof(uint64_t));
        // reader.read((char*)&ncols, sizeof(uint64_t));
        reader.read((char*)&nrows_32, sizeof(uint32_t));
        reader.read((char*)&ncols_32, sizeof(uint32_t));
        nrows = nrows_32;
        ncols = ncols_32;

        return true;
    } else {
        LOG_KNOWHERE_ERROR_ << "Could not open file: " << bin_file;
        return false;
    }
}

template <typename T>
inline bool
write_output_bin(const std::string& bin_file, T* data, uint64_t npts, uint64_t ndim) {
    std::ofstream writer(bin_file, std::ios::binary | std::ios::out);
    if (!writer.fail() && writer.is_open()) {
        uint32_t npts_32 = (uint32_t)npts;
        uint32_t ndim_32 = (uint32_t)ndim;
        writer.write((char*)&npts_32, sizeof(uint32_t));
        writer.write((char*)&ndim_32, sizeof(uint32_t));

        // writer.write((char*)&npts, sizeof(uint64_t));
        // writer.write((char*)&ndim, sizeof(uint64_t));
        LOG_KNOWHERE_INFO_ << "bin: #pts = " << npts << ", #dim = " << ndim
                           << ", size = " << npts * ndim * sizeof(T) + 2 * sizeof(uint32_t) << "B";
        writer.write((char*)data, npts * ndim * sizeof(T));
        writer.close();
        return true;
    } else {
        LOG_KNOWHERE_ERROR_ << "Could not write file: " << bin_file;
        return false;
    }
}
}  // namespace

template <typename VecT = float>
class KMeans {
 public:
    KMeans(size_t K, size_t dim, bool verbose = true) : dim_(dim), n_centroids_(K), verbose_(verbose) {
    }

    void
    fit(const VecT* vecs, size_t n, size_t max_iter = 10, uint32_t random_state = 0, std::string_view init = "random",
        std::string_view algorithm = "lloyd");

    void
    computeClosestCentroid(const VecT* vecs, size_t n, uint32_t* closest_centroid, float* closest_centroid_distance);

    std::unique_ptr<VecT[]>&
    get_centroids() {
        return centroids_;
    }

    std::unique_ptr<uint32_t[]>&
    get_cluster_id_mapping() {
        return cluster_id_mapping_;
    }

    ~KMeans() {
    }

 private:
    size_t dim_, n_centroids_;
    std::unique_ptr<VecT[]> centroids_;
    std::unique_ptr<uint32_t[]> cluster_id_mapping_;

    bool verbose_ = true;

    void
    initRandom(const VecT* train_data, size_t n_train, uint32_t random_state);

    void
    initKMeanspp(const VecT* train_data, size_t n_train, uint32_t random_state);

    float
    lloyds_iter(const VecT* train_data, std::vector<std::vector<uint32_t>>& closest_docs, uint32_t* closest_centroids,
                float* closest_centroid_distancessize_t, size_t n_train, uint32_t random_state,
                bool compute_residual = false);

    void
    split_clusters(std::vector<int>& hassign, size_t n_train, uint32_t random_state);

    void
    elkan_L2(const VecT* x, const VecT* y, size_t d, size_t nx, size_t ny, uint32_t* ids, float* val);

    void
    exhaustive_L2sqr_blas(const VecT* x, const VecT* y, size_t d, size_t nx, size_t ny, uint32_t* ids, float* val);
};

template <typename VecT>
expected<DataSetPtr>
ClusteringMajorCompaction(const std::vector<std::string>& file_paths, const std::string& output_path_prefix,
                          int num_clusters, std::shared_ptr<FileManager> file_manager);

}  // namespace knowhere::kmeans
