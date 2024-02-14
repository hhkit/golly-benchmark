//
// Created by jiashuai on 17-10-6.
//

#ifndef THUNDERSVM_ONECLASS_SVC_H
#define THUNDERSVM_ONECLASS_SVC_H

#include "svmmodel.h"

class OneClassSVC : public SvmModel {
public:
    OneClassSVC() : SvmModel() {
        param.svm_type = SvmParam::ONE_CLASS;
    }

    void train(DataSet dataset, SvmParam param) override;

    void save_to_file(string path) override;

    void load_from_file(string path) override;

    vector<real> predict(const DataSet::node2d &instances, int batch_size) override;

};

#endif //THUNDERSVM_ONECLASS_SVC_H
