# frozen_string_literal: true

require "cli/parser"
require "utils/spdx"

module Homebrew
  module_function

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
    args = update_license_data_args.parse
    ohai "Updating SPDX license data..."

    SPDX.download_latest_license_data!

    Homebrew.failed = system("git", "diff", "--stat", "--exit-code", SPDX::JSON_PATH) if args.fail_if_not_changed?

    return unless args.commit?

    ohai "git add"
    safe_system "git", "add", SPDX::JSON_PATH
    ohai "git commit"
    system "git", "commit", "--message", "data/spdx.json: update to #{latest_tag}"
  end
end
