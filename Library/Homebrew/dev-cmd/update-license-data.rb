# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  module_function

  SPDX_PATH = (HOMEBREW_LIBRARY_PATH/"data/spdx.json").freeze
  SPDX_API_URL = "https://api.github.com/repos/spdx/license-list-data/releases/latest"

  def update_license_data_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `update-license-data` [<options>]

         Update SPDX license data in the Homebrew repository.
      EOS
      switch "--fail-if-not-changed",
             description: "Return a failing status code if current license data's version is the same as " \
                          "the upstream. This can be used to notify CI when the SPDX license data is out of date."
      switch "--commit",
             description: "Commit changes to the SPDX license data."
      max_named 0
    end
  end

  def update_license_data
    update_license_data_args.parse
    ohai "Updating SPDX license data..."

    latest_tag = GitHub.open_api(SPDX_API_URL)["tag_name"]
    data_url = "https://raw.githubusercontent.com/spdx/license-list-data/#{latest_tag}/json/licenses.json"
    curl_download(data_url, to: SPDX_PATH, partial: false)

    Homebrew.failed = system("git", "diff", "--stat", "--exit-code", SPDX_PATH) if args.fail_if_not_changed?

    return unless args.commit?

    ohai "git add"
    safe_system "git", "add", SPDX_PATH
    ohai "git commit"
    system "git", "commit", "--message", "data/spdx.json: update to #{latest_tag}"
  end
end
