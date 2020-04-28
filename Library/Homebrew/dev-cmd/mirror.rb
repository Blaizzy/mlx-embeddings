# frozen_string_literal: true

require "bintray"
require "cli/parser"

module Homebrew
  module_function

  def mirror_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `mirror` <formula>

        Reupload the stable URL of a formula to Bintray for use as a mirror.
      EOS
      flag "--bintray-org=",
           description: "Upload to the specified Bintray organisation (default: homebrew)."
      switch :verbose
      switch :debug
      hide_from_man_page!
      min_named :formula
    end
  end

  def mirror
    mirror_args.parse

    bintray_org = args.bintray_org || "homebrew"
    bintray_repo = "mirror"

    bintray = Bintray.new(org: bintray_org)

    args.formulae.each do |f|
      bintray_package = Utils::Bottles::Bintray.package f.name

      unless bintray.package_exists?(repo: bintray_repo, package: bintray_package)
        bintray.create_package repo: bintray_repo, package: bintray_package
      end

      downloader = f.downloader

      downloader.fetch

      filename = downloader.basename

      destination_url = "https://dl.bintray.com/#{bintray_org}/#{bintray_repo}/#{filename}"
      ohai "Uploading to #{destination_url}"

      bintray.upload(
        downloader.cached_location,
        repo:        bintray_repo,
        package:     bintray_package,
        version:     f.pkg_version,
        sha256:      f.stable.checksum,
        remote_file: filename,
      )
      bintray.publish(repo: bintray_repo, package: bintray_package, version: f.pkg_version)
      ohai "Mirrored #{filename}!"
    end
  end
end
