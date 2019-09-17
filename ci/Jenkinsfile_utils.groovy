// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// initialize source codes
def init_git(git_sha = '') {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        sh 'git clean -xdff'
        sh 'git reset --hard'
        sh 'git submodule update --init --recursive'
        sh 'git submodule foreach --recursive git clean -ffxd'
        sh 'git submodule foreach --recursive git reset --hard'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
      sleep 2
    }
  }
}

def init_git_win() {
  deleteDir()
  retry(5) {
    try {
      // Make sure wait long enough for api.github.com request quota. Important: Don't increase the amount of
      // retries as this will increase the amount of requests and worsen the throttling
      timeout(time: 15, unit: 'MINUTES') {
        checkout scm
        bat 'git clean -xdff'
        bat 'git reset --hard'
        bat 'git submodule update --init --recursive'
        bat 'git submodule foreach --recursive git clean -ffxd'
        bat 'git submodule foreach --recursive git reset --hard'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes with ${exc}"
      sleep 2
    }
  }
}

// pack libraries for later use
def pack_lib(name, libs, include_gcov_data = false) {
  sh returnStatus: true, script: """
set +e
echo "Packing ${libs} into ${name}"
for i in \$(echo ${libs} | sed -e 's/,/ /g'); do md5sum \$i; done
return 0
"""
  stash includes: libs, name: name

  if (include_gcov_data) {
    // Store GCNO files that are required for GCOV to operate during runtime
    sh "find . -name '*.gcno'"
    stash name: "${name}_gcov_data", includes: "**/*.gcno"
  }
}

// unpack libraries saved before
def unpack_and_init(name, libs, include_gcov_data = false, git_sha = '') {
  init_git(git_sha)
  unstash name
  sh returnStatus: true, script: """
set +e
echo "Unpacked ${libs} from ${name}"
for i in \$(echo ${libs} | sed -e 's/,/ /g'); do md5sum \$i; done
return 0
"""
  if (include_gcov_data) {
    // Restore GCNO files that are required for GCOV to operate during runtime
    unstash "${name}_gcov_data"
  }
}

def get_jenkins_master_url() {
    return env.BUILD_URL.split('/')[2].split(':')[0]
}

def get_git_commit_hash() {
  lastCommitMessage = sh (script: "git log -1 --pretty=%B", returnStdout: true)
  lastCommitMessage = lastCommitMessage.trim()
  if (lastCommitMessage.startsWith("Merge commit '") && lastCommitMessage.endsWith("' into HEAD")) {
      // Merge commit applied by Jenkins, skip that commit
      git_commit_hash = sh (script: "git rev-parse @~", returnStdout: true)
  } else {
      git_commit_hash = sh (script: "git rev-parse @", returnStdout: true)
  }
  return git_commit_hash
}

def publish_test_coverage() {
    // CodeCovs auto detection has trouble with our CIs PR validation due the merging strategy
    git_commit_hash = get_git_commit_hash()
   
    if (env.CHANGE_ID) {
      // PR execution
      codecovArgs = "-B ${env.CHANGE_TARGET} -C ${git_commit_hash} -P ${env.CHANGE_ID}"
    } else {
      // Branch execution
      codecovArgs = "-B ${env.BRANCH_NAME} -C ${git_commit_hash}"
    }

    // To make sure we never fail because test coverage reporting is not available
    // Fall back to our own copy of the bash helper if it failed to download the public version
    sh "(curl --retry 10 -s https://codecov.io/bash | bash -s - ${codecovArgs}) || (curl --retry 10 -s https://s3-us-west-2.amazonaws.com/mxnet-ci-prod-slave-data/codecov-bash.txt | bash -s - ${codecovArgs}) || true"
}

def collect_test_results_unix(original_file_name, new_file_name) {
    if (fileExists(original_file_name)) {
        // Rename file to make it distinguishable. Unfortunately, it's not possible to get STAGE_NAME in a parallel stage
        // Thus, we have to pick a name manually and rename the files so that they can be stored separately.
        sh 'cp ' + original_file_name + ' ' + new_file_name
        archiveArtifacts artifacts: new_file_name
    }
}

def collect_test_results_windows(original_file_name, new_file_name) {
    // Rename file to make it distinguishable. Unfortunately, it's not possible to get STAGE_NAME in a parallel stage
    // Thus, we have to pick a name manually and rename the files so that they can be stored separately.
    if (fileExists(original_file_name)) {
        bat 'xcopy ' + original_file_name + ' ' + new_file_name + '*'
        archiveArtifacts artifacts: new_file_name
    }
}


def docker_run(platform, function_name, use_nvidia, shared_mem = '500m', env_vars = "") {
  def command = "ci/build.py %ENV_VARS% --docker-registry ${env.DOCKER_CACHE_REGISTRY} %USE_NVIDIA% --platform %PLATFORM% --docker-build-retries 3 --shm-size %SHARED_MEM% /work/runtime_functions.sh %FUNCTION_NAME%"
  command = command.replaceAll('%ENV_VARS%', env_vars.length() > 0 ? "-e ${env_vars}" : '')
  command = command.replaceAll('%USE_NVIDIA%', use_nvidia ? '--nvidiadocker' : '')
  command = command.replaceAll('%PLATFORM%', platform)
  command = command.replaceAll('%FUNCTION_NAME%', function_name)
  command = command.replaceAll('%SHARED_MEM%', shared_mem)

  sh command
}

// Allow publishing to GitHub with a custom context (the status shown under a PR)
// Credit to https://plugins.jenkins.io/github
def get_repo_url() {
  checkout scm
  return sh(returnStdout: true, script: "git config --get remote.origin.url").trim()
}

def update_github_commit_status(state, message) {
  node(NODE_UTILITY) {
    // NOTE: https://issues.jenkins-ci.org/browse/JENKINS-39482
    //The GitHubCommitStatusSetter requires that the Git Server is defined under 
    //*Manage Jenkins > Configure System > GitHub > GitHub Servers*. 
    //Otherwise the GitHubCommitStatusSetter is not able to resolve the repository name 
    //properly and you would see an empty list of repos:
    //[Set GitHub commit status (universal)] PENDING on repos [] (sha:xxxxxxx) with context:test/mycontext
    //See https://cwiki.apache.org/confluence/display/MXNET/Troubleshooting#Troubleshooting-GitHubcommit/PRstatusdoesnotgetpublished

    echo "Publishing commit status..."

    repoUrl = get_repo_url()
    echo "repoUrl=${repoUrl}"

    commitSha = get_git_commit_hash()
    echo "commitSha=${commitSha}"
    
    context = get_github_context()
    echo "context=${context}"

    // a few attempts need to be made: https://github.com/apache/incubator-mxnet/issues/11654
    for (int attempt = 1; attempt <= 3; attempt++) {
      echo "Sending GitHub status attempt ${attempt}..."

      step([
        $class: 'GitHubCommitStatusSetter',
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: repoUrl],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: context],
        commitShaSource: [$class: "ManuallyEnteredShaSource", sha: commitSha],
        statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: "${env.RUN_DISPLAY_URL}"],
        errorHandlers: [[$class: 'ShallowAnyErrorHandler']],
        statusResultSource: [
          $class: 'ConditionalStatusResultSource',
          results: [[$class: "AnyBuildResult", message: message, state: state]]
        ]
      ])

      if (attempt <= 2) {
        sleep 1
      }
    }

    echo "Publishing commit status done."

  }
}

def get_github_context() {
  // Since we use multi-branch pipelines, Jenkins appends the branch name to the job name
  if (env.BRANCH_NAME) {
    short_job_name = JOB_NAME.substring(0, JOB_NAME.lastIndexOf('/')) 
  } else {
    short_job_name = JOB_NAME
  }
  
  return "ci/jenkins/${short_job_name}"
}

def parallel_stage(stage_name, steps) {
    // Allow to pass an array of steps that will be executed in parallel in a stage
    new_map = [:]
    
    for (def step in steps) {
        new_map = new_map << step
    }
    
    stage(stage_name) {
      parallel new_map
    }
}

def assign_node_labels(args) {
  // This function allows to assign instance labels to the generalized placeholders. 
  // This serves two purposes:
  // 1. Allow generalized placeholders (e.g. NODE_WINDOWS_CPU) in the job definition
  //    in order to abstract away the underlying node label. This allows to schedule a job
  //    onto a different node for testing or security reasons. This could be, for example,
  //    when you want to test a new set of slaves on separate labels or when a job should
  //    only be run on restricted slaves
  // 2. Restrict the allowed job types within a Jenkinsfile. For example, a UNIX-CPU-only
  //    Jenkinsfile should not allowed access to Windows or GPU instances. This prevents
  //    users from just copy&pasting something into an existing Jenkinsfile without
  //    knowing about the limitations.
  NODE_LINUX_CPU = args.linux_cpu
  NODE_LINUX_GPU = args.linux_gpu
  NODE_LINUX_GPU_P3 = args.linux_gpu_p3
  NODE_WINDOWS_CPU = args.windows_cpu
  NODE_WINDOWS_GPU = args.windows_gpu
  NODE_UTILITY = args.utility
}

def check_only_doc_tutorials_changes(){
  checkout scm
  is_doc_tutorials = sh (returnStdout: true, script: """ 
  set +e
  c=`git --no-pager diff --name-only HEAD master`
  stringarray=${c}
  for i in  "${stringarray}"
  do 
    if [[ $i == docs/* ]] || [[ $i == tests/nightly* ]] || [[ $i == ci/* ]] ;
    then
      continue
    else
      echo "false"
      return 0
    fi
  done
  echo "true"    
  return 0
""")
  lines = is_doc_tutorials.trim()
  return lines == "true"
}
  
def main_wrapper(args) {
  // Main Jenkinsfile pipeline wrapper handler that allows to wrap core logic into a format
  // that supports proper failure handling
  // args:
  // - core_logic: Jenkins pipeline containing core execution logic
  // - failure_handler: Failure handler
  
  // assign any caught errors here
  err = null
  try {
    if(check_only_doc_tutorials_changes()){   
      node(NODE_UTILITY) {
        sh "echo skipping tests because only docs/tutorials change"
        currentBuild.result = "SUCCESS"
        update_github_commit_status('SUCCESS', 'Skipped as only doc and tutorials changes')
      }
    } else {
    
    update_github_commit_status('PENDING', 'Job has been enqueued')
    args['core_logic']()

    // set build status to success at the end
    currentBuild.result = "SUCCESS"
    update_github_commit_status('SUCCESS', 'Job succeeded')
    }
  } catch (caughtError) {
    node(NODE_UTILITY) {
      sh "echo caught ${caughtError}"
      err = caughtError
      currentBuild.result = "FAILURE"
      update_github_commit_status('FAILURE', 'Job failed')
    }
  } finally {
    node(NODE_UTILITY) {
      // Call failure handler
      args['failure_handler']()

      // Clean workspace to reduce space requirements
      cleanWs()

      // Remember to rethrow so the build is marked as failing
      if (err) {
        throw err
      }
    }
  }
}

return this
