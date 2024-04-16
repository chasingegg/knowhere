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

#include "knowhere/clustering/clustering_factory.h"

namespace knowhere {

template <typename DataType>
expected<Clustering<ClusteringNode>>
ClusteringFactory::Create(const std::string& name, const int32_t& version, const Object& object) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetIndexKey<DataType>(name);
    if (func_mapping_.find(name) == func_mapping_.end()) {
        LOG_KNOWHERE_ERROR_ << "failed to find index " << name << " in factory";
        return expected<Clustering<ClusteringNode>>::Err(Status::invalid_index_error, "index not supported");
    }
    LOG_KNOWHERE_INFO_ << "use name " << name << " to create knowhere index " << name << " with version " << version;
    auto fun_map_v = (FunMapValue<Clustering<ClusteringNode>>*)(func_mapping_[name].get());

    return fun_map_v->fun_value(version, object);
}

template <typename DataType>
const ClusteringFactory&
ClusteringFactory::Register(const std::string& name,
                            std::function<Clustering<ClusteringNode>(const int32_t&, const Object&)> func) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetIndexKey<DataType>(name);
    assert(func_mapping_.find(key) == func_mapping_.end());
    func_mapping_[key] = std::make_unique<FunMapValue<Clustering<ClusteringNode>>>(func);
    return *this;
}

ClusteringFactory&
ClusteringFactory::Instance() {
    static ClusteringFactory factory;
    return factory;
}

ClusteringFactory::ClusteringFactory() {
}

ClusteringFactory::FuncMap&
ClusteringFactory::MapInstance() {
    static FuncMap func_map;
    return func_map;
}

}  // namespace knowhere
   //
template knowhere::expected<knowhere::Clustering<knowhere::ClusteringNode>>
knowhere::ClusteringFactory::Create<knowhere::fp32>(const std::string&, const int32_t&, const Object&);
template const knowhere::ClusteringFactory&
knowhere::ClusteringFactory::Register<knowhere::fp32>(
    const std::string&, std::function<knowhere::Clustering<knowhere::ClusteringNode>(const int32_t&, const Object&)>);
