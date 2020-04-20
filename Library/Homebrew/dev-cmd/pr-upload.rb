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
    end
  end

  def pr_upload
    pr_upload_args.parse

    bintray_org = args.bintray_org || "homebrew"
    bintray = Bintray.new(org: bintray_org)

    if args.dry_run?
      puts "brew bottle --merge --write #{Dir["*.json"].join " "}"
    else
      system HOMEBREW_BREW_FILE, "bottle", "--merge", "--write", *Dir["*.json"]
    end

    if args.dry_run?
      puts "Upload bottles described by these JSON files to Bintray:\n  #{Dir["*.json"].join("\n  ")}"
    else
      bintray.upload_bottle_json Dir["*.json"], publish_package: !args.no_publish?
    end
  end
end
