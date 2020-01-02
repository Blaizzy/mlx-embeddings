# frozen_string_literal: true

require "sandbox"
require "formula_installer"
require "cli/parser"

module Homebrew
  module_function

  def postinstall_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `postinstall` <formula>

        Rerun the post-install steps for <formula>.
      EOS
      switch :force
      switch :verbose
      switch :debug
    end
  end

  def postinstall
    postinstall_args.parse

    raise KegUnspecifiedError if args.remaining.empty?

    Homebrew.args.resolved_formulae.each do |f|
      ohai "Postinstalling #{f}"
      fi = FormulaInstaller.new(f)
      fi.post_install
    end
  end
end
