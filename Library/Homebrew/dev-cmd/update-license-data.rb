# typed: true
# frozen_string_literal: true

require "cli/parser"
require "utils/spdx"
require "system_command"

module Homebrew
  extend T::Sig
  include SystemCommand::Mixin

  module_function

  sig { returns(CLI::Parser) }
  def update_license_data_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Update SPDX license data in the Homebrew repository.
      EOS
      switch "--fail-if-not-changed",
             hidden:      true,
             description: "Return a failing status code if current license data's version is the same as " \
                          "the upstream. This can be used to notify CI when the SPDX license data is out of date."

      named_args :none
    end
  end

  def update_license_data
    args = update_license_data_args.parse
    odeprecated "brew update-license-data --fail-if-not-changed" if args.fail_if_not_changed?

    SPDX.download_latest_license_data!
    diff = system_command "git", args: [
      "-C", HOMEBREW_REPOSITORY, "diff", "--exit-code", SPDX::DATA_PATH
    ]
    if diff.status.success?
      ofail "No changes to SPDX license data."
    else
      puts "SPDX license data updated."
    end
  end
end
