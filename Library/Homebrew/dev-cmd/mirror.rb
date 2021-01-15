# typed: true
# frozen_string_literal: true

require "bintray"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def mirror_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Reupload the stable URL of a formula to Bintray for use as a mirror.
      EOS
      flag   "--bintray-org=",
             description: "Upload to the specified Bintray organisation (default: `homebrew`)."
      flag   "--bintray-repo=",
             description: "Upload to the specified Bintray repository (default: `mirror`)."
      switch "--no-publish",
             description: "Upload to Bintray, but don't publish."

      named_args :formula, min: 1
    end
  end

  def mirror
    args = mirror_args.parse

    bintray_org = args.bintray_org || "homebrew"
    bintray_repo = args.bintray_repo || "mirror"

    bintray = Bintray.new(org: bintray_org)

    args.named.to_formulae.each do |formula|
      mirror_url = bintray.mirror_formula(formula, repo: bintray_repo, publish_package: !args.no_publish?)
      ohai "Mirrored #{formula.full_name} to #{mirror_url}!"
    end
  end
end
