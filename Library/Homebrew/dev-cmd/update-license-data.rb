# typed: true
# frozen_string_literal: true

require "cli/parser"
require "utils/spdx"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def update_license_data_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Update SPDX license data in the Homebrew repository.
      EOS
      switch "--fail-if-not-changed",
             description: "Return a failing status code if current license data's version is the same as " \
                          "the upstream. This can be used to notify CI when the SPDX license data is out of date."

      named_args :none
    end
  end

  def update_license_data
    args = update_license_data_args.parse
    ohai "Updating SPDX license data..."

    SPDX.download_latest_license_data!
    return unless args.fail_if_not_changed?

    Homebrew.failed = system("git", "diff", "--stat", "--exit-code", SPDX::DATA_PATH)
  end
end
