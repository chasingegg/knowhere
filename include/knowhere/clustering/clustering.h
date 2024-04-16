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

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "knowhere/binaryset.h"
#include "knowhere/clustering/clustering_node.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"

namespace knowhere {
template <typename T1>
class Clustering {
 public:
    template <typename T2>
    friend class Clustering;

    Clustering() : node(nullptr) {
    }

    template <typename... Args>
    static Clustering<T1>
    Create(Args&&... args) {
        return Clustering(new (std::nothrow) T1(std::forward<Args>(args)...));
    }

    Clustering(const Clustering<T1>& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        idx.node->IncRef();
        node = idx.node;
    }

    Clustering(Clustering<T1>&& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    template <typename T2>
    Clustering(const Clustering<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        idx.node->IncRef();
        node = idx.node;
    }

    template <typename T2>
    Clustering(Clustering<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    template <typename T2>
    Clustering<T1>&
    operator=(const Clustering<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        if (idx.node == nullptr) {
            node = nullptr;
            return *this;
        }
        node = idx.node;
        node->IncRef();
        return *this;
    }

    template <typename T2>
    Clustering<T1>&
    operator=(Clustering<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        node = idx.node;
        idx.node = nullptr;
        return *this;
    }

    T1*
    Node() {
        return node;
    }

    const T1*
    Node() const {
        return node;
    }

    ~Clustering() {
        if (node == nullptr)
            return;
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

 private:
    Clustering(T1* node) : node(node) {
        static_assert(std::is_base_of<ClusteringNode, T1>::value);
    }

    T1* node;
};

}  // namespace knowhere

#endif /* CLUSTERING_H */
