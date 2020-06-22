# frozen_string_literal: true

require "commands"
require "cli/parser"
require "open-uri"
require "json"

module Homebrew
  module_function
  SPDX_FOLDER_PATH = (HOMEBREW_LIBRARY_PATH/"data").freeze
  FILE_NAME = "spdx.json".freeze
  NEW_FILE_NAME = "spdx_new.json".freeze
  SPDX_DATA_URL = "https://raw.githubusercontent.com/spdx/license-list-data/master/json/licenses.json"

  def update_license_data_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `update_license_data` <cmd>

        Update SPDX license data in the Homebrew repository.
      EOS
      switch "--fail-if-outdated",
             description: "Return a failing status code if current license data's version is different from the upstream. This "\
                          "can be used to notify CI when the SPDX license data is out of date."

      switch "--do-not-replace",
             description: "Flags out discrepancy between local and upstream versions, but does not replace"

      max_named 0
    end
  end

  def update_license_data
    update_license_data_args.parse
    p args
    curr_spdx_hash = File.open(SPDX_FOLDER_PATH/FILE_NAME, 'r') do |f|
      JSON.parse(f.read)
    end
    puts "Fetching newest version of SPDX License data..."
    updated_spdx_string = open(SPDX_DATA_URL) do |json|
       json.read
    end

    updated_spdx_hash = JSON.parse(updated_spdx_string)
    if curr_spdx_hash["licenseListVersion"] != updated_spdx_hash["licenseListVersion"]

      puts "Current version is #{curr_spdx_hash["licenseListVersion"]} but newest version is #{updated_spdx_hash["licenseListVersion"]}"
      unless args.do_not_replace?
        puts "Updating existing licences data file..."
        File.open(SPDX_FOLDER_PATH/FILE_NAME, "wb") do |file|
            file.write(updated_spdx_string)
        end
      end
      Homebrew.failed = !!args.fail_if_outdated
    else
      puts "Current version of license data is updated. No change required"
    end
  end
  end
