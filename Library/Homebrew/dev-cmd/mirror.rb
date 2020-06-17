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

    bintray = Bintray.new(org: bintray_org)

    args.formulae.each do |formula|
      mirror_url = bintray.mirror_formula(formula)
      ohai "Mirrored #{formula.full_name} to #{mirror_url}!"
    end
  end
end
