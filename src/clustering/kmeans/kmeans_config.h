// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef KMEANS_CONFIG_H
#define KMEANS_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class KmeansConfig : public BaseConfig {
 public:
    CFG_INT num_clusters;
    CFG_INT max_iter;
    CFG_INT random_state;
    KNOHWERE_DECLARE_CONFIG(KmeansConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(num_clusters)
            .description("num of clusters")
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_clustering();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_iter)
            .description("max iterations of kmeans clustering")
            .set_default(10)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_clustering();
        KNOWHERE_CONFIG_DECLARE_FIELD(random_state)
            .description("random state of kmeans clustering")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_INT::value_type>::max())
            .for_clustering();
    }
};

}  // namespace knowhere

#endif /* KMEANS_CONFIG_H */
