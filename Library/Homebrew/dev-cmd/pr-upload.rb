# frozen_string_literal: true

require "cli/parser"
require "bintray"

module Homebrew
  module_function

  def pr_upload_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pr-upload` [<options>]

        Apply the bottle commit and publish bottles to Bintray.
      EOS
      switch "--no-publish",
             description: "Apply the bottle commit and upload the bottles, but don't publish them."
      switch "--dry-run", "-n",
             description: "Print what would be done rather than doing it."
      flag "--bintray-org=",
           description: "Upload to the specified Bintray organisation (default: homebrew)."
      flag "--root-url=",
           description: "Use the specified <URL> as the root of the bottle's URL instead of Homebrew's default."
    end
  end

  def pr_upload
    pr_upload_args.parse

    bintray_org = args.bintray_org || "homebrew"
    bintray = Bintray.new(org: bintray_org)

    bottle_args = ["bottle", "--merge", "--write"]
    bottle_args << "--root-url=#{args.root_url}" if args.root_url
    odie "No JSON files found in the current working directory" if Dir["*.json"].empty?
    bottle_args += Dir["*.json"]

    if args.dry_run?
      puts "brew #{bottle_args.join " "}"
    else
      system HOMEBREW_BREW_FILE, *bottle_args
    end

    if args.dry_run?
      puts "Upload bottles described by these JSON files to Bintray:\n  #{Dir["*.json"].join("\n  ")}"
    else
      bintray.upload_bottle_json Dir["*.json"], publish_package: !args.no_publish?
    end
  end
end
