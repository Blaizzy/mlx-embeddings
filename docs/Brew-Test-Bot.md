---
logo: https://brew.sh/assets/img/brewtestbot.png
image: https://brew.sh/assets/img/brewtestbot.png
---

# Brew Test Bot

`brew test-bot` is the name for the automated review and testing system funded
by [our Kickstarter in 2013](https://www.kickstarter.com/projects/homebrew/brew-test-bot).

It comprises three Mac Pros hosting virtual machines that run the
[`test-bot.rb`](https://github.com/Homebrew/homebrew-test-bot/) external
command to perform automated testing of commits to the master branch, pull
requests and custom builds requested by maintainers.

## Pull Requests

The bot automatically builds pull requests and updates their status depending
on the result of the job.

For example, a job which has been queued but not yet completed will have a
section in the pull request that looks like this:

![Triggered Pull Request](assets/img/docs/brew-test-bot-triggered-pr.png)

---

A failed build looks like this:

![Failed Pull Request](assets/img/docs/brew-test-bot-failed-pr.png)

---

A passed build looks like this:

![Passed Pull Request](assets/img/docs/brew-test-bot-passed-pr.png)

---

On failed or passed builds you can click the "Details" link to view the result
in GitHub Actions.

