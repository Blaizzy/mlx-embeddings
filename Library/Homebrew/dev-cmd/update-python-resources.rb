# frozen_string_literal: true

require "cli/parser"
require "utils/pypi"

module Homebrew
  module_function

  def update_python_resources_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `update-python-resources` [<options>] <formula>

        Update versions for PyPI resource blocks in <formula>.
      EOS
      switch "-p", "--print-only",
             description: "Print the updated resource blocks instead of changing <formula>."
      switch "-s", "--silent",
             description: "Suppress any output."
      switch "--ignore-non-pypi-packages",
             description: "Don't fail if <formula> is not a PyPI package."
      flag "--version=",
           description: "Use the specified <version> when finding resources for <formula>. "\
                        "If no version is specified, the current version for <formula> will be used."
      min_named :formula
    end
  end

  def update_python_resources
    args = update_python_resources_args.parse

    args.named.to_formulae.each do |formula|
      PyPI.update_python_resources! formula, args.version, print_only: args.print_only?, silent: args.silent?,
                                    ignore_non_pypi_packages: args.ignore_non_pypi_packages?
    end
  end
end
