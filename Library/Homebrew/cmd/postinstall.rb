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

      min_named :keg
    end
  end

  def postinstall
    args = postinstall_args.parse

    args.named.to_resolved_formulae.each do |f|
      ohai "Postinstalling #{f}"
      fi = FormulaInstaller.new(f, debug: args.debug?, quiet: args.quiet?, verbose: args.verbose?)
      fi.post_install
    end
  end
end
