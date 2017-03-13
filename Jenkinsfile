def mx_lib = 'lib/libmxnet.so'
def mx_run = 'tests/ci_build/ci_build.sh'

def pack_lib(name, mx_lib) {
  sh """
echo "Packing ${mx_lib} into ${name}"
md5sum ${mx_lib}
"""
  stash includes: mx_lib, name: name
}

def unpack_lib(name, mx_lib) {
  unstash name
  sh """
echo "Unpacked ${mx_lib} from ${name}"
md5sum ${mx_lib}
"""
}

stage("Sanity Check") {
  node {
    ws('workspace/sanity') {
      checkout scm
      sh 'git submodule update --init'
      sh '${mx_run} lint make cpplint'
      sh '${mx_run} lint make rcpplint'
      sh '${mx_run} lint make jnilint'
      sh '${mx_run} lint make pylint'
    }
  }
}


stage('Build') {
  parallel 'CPU': {
    node {
      ws('workspace/build-cpu') {
        checkout scm
        sh 'git submodule update --init'
        sh "${mx_run} cpu make -j\$(nproc) USE_BLAS=openblas"
        pack_lib 'cpu', mx_lib
      }
    }
  },
  'GPU: CUDA7.5+cuDNN5': {
    node('GPU') {
      ws('workspace/build-gpu') {
        checkout scm
        sh 'git submodule update --init'
        sh "${mx_run} gpu make -j\$(nproc) USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
        pack_lib 'gpu', mx_lib
      }
    }
  }
}

stage('Unit Test') {
  parallel 'Python2/3: CPU': {
    node {
      ws('workspace/ut-python-cpu') {
        checkout scm
        sh 'git submodule update --init'
        unpack_lib 'cpu', mx_lib
        sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests --verbose tests/python/unittest'"
        sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests3 --verbose tests/python/unittest'"
      }
    }
  },
  'Python2/3: GPU': {
    node('GPU') {
      ws('workspace/ut-python-gpu') {
        checkout scm
        sh 'git submodule update --init'
        unpack_lib 'gpu', mx_lib
        sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests --verbose tests/python/unittest'"
        sh "${mx_run} cpu 'PYTHONPATH=./python/ nosetests3 --verbose tests/python/unittest'"
      }
    }
  },
  'Scala: CPU': {
    node {
      ws('workspace/ut-scala-cpu') {
        checkout scm
        sh 'git submodule update --init'
        unpack_lib 'cpu', mx_lib
        sh "${mx_run} cpu make scalapkg"
        sh "${mx_run} cpu make scalatest"
      }
    }
  }
}
