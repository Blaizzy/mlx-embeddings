workflow "Push" {
  on = "push"
  resolves = ["Generate rubydoc.brew.sh"]
}

action "Generate rubydoc.brew.sh" {
  uses = "docker://ruby:latest"
  runs = ".github/main.workflow.sh"
  secrets = ["RUBYDOC_DEPLOY_KEY"]
}
