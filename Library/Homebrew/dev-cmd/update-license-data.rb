# frozen_string_literal: true

require "commands"
require "cli/parser"
require "open-uri"
require "json"

module Homebrew
  module_function

    SPDX_FOLDER_PATH = (HOMEBREW_LIBRARY_PATH/"data").freeze
    FILE_NAME = "spdx.json"
    SPDX_DATA_URL = "https://raw.githubusercontent.com/spdx/license-list-data/master/json/licenses.json"

    def update_license_data_args
      Homebrew::CLI::Parser.new do
        usage_banner <<~EOS
          `update_license_data` <cmd>

          Update SPDX license data in the Homebrew repository.
      EOS
        switch "--fail-if-changed",
               description: "Return a failing status code if current license data's version is different from the upstream. This "\
                            "can be used to notify CI when the SPDX license data is out of date."

        max_named 0
      end
    end

    def update_license_data
      update_license_data_args.parse
      puts "Fetching newest version of SPDX License data..."
      open(SPDX_DATA_URL) do |json|
        File.open(SPDX_FOLDER_PATH/FILE_NAME, "wb") do |file|
          file.write(json.read)
        end
      end

      if args.fail_if_changed?
        Homebrew.failed = true
          system("git diff --stat --exit-code #{FILE_NAME}")
      end
    end
  end
