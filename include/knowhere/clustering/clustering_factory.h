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

#ifndef CLUSTERING_FACTORY_H
#define CLUSTERING_FACTORY_H

#include <functional>
#include <string>
#include <unordered_map>

#include "knowhere/clustering/clustering.h"
#include "knowhere/utils.h"

namespace knowhere {
class ClusteringFactory {
 public:
    template <typename DataType>
    expected<Clustering<ClusteringNode>>
    Create(const std::string& name, const Object& object = nullptr);
    template <typename DataType>
    const ClusteringFactory&
    Register(const std::string& name, std::function<Clustering<ClusteringNode>(const Object&)> func);
    static ClusteringFactory&
    Instance();

 private:
    struct FunMapValueBase {
        virtual ~FunMapValueBase() = default;
    };
    template <typename T1>
    struct FunMapValue : FunMapValueBase {
     public:
        FunMapValue(std::function<T1(const Object&)>& input) : fun_value(input) {
        }
        std::function<T1(const Object&)> fun_value;
    };
    typedef std::map<std::string, std::unique_ptr<FunMapValueBase>> FuncMap;
    ClusteringFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CLUSTERING_CONCAT(x, y) clustering_factory_ref_##x##y
#define KNOWHERE_CLUSTEDRING_REGISTER_GLOBAL(name, func, data_type)        \
    const ClusteringFactory& KNOWHERE_CLUSTERING_CONCAT(name, data_type) = \
        ClusteringFactory::Instance().Register<data_type>(#name, func)
#define KNOWHERE_CLUSTERING_SIMPLE_REGISTER_GLOBAL(name, clustering_node, data_type, ...)       \
    KNOWHERE_CLUSTEDRING_REGISTER_GLOBAL(                                                       \
        name,                                                                                   \
        (static_cast<Clustering<clustering_node<data_type, ##__VA_ARGS__>> (*)(const Object&)>( \
            &Clustering<clustering_node<data_type, ##__VA_ARGS__>>::Create)),                   \
        data_type)
}  // namespace knowhere

#endif /* CLUSTERING_FACTORY_H */
