# typed: true
# frozen_string_literal: true

require "cli/parser"
require "formula"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def generate_formula_api_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Generates Formula API data files for formulae.brew.sh.
      EOS

      named_args :none
    end
  end

  JSON_TEMPLATE = <<~EOS
    ---
    layout: formula_json
    ---
    {{ content }}
  EOS

  def html_template(title)
    <<~EOS
      ---
      title: #{title}
      layout: formula
      redirect_from: /formula-linux/$TITLE
      ---
      {{ content }}
    EOS
  end

  def generate_formula_api
    generate_formula_api_args.parse

    tap = Tap.fetch("homebrew/core")

    directories = ["_data/formula", "api/formula", "formula"]
    FileUtils.rm_rf directories + ["_data/formula_canonical.json"]
    FileUtils.mkdir_p directories

    Formulary.enable_factory_cache!

    tap.formula_names.each do |name|
      formula = Formulary.factory(name)
      name = formula.name
      json = JSON.pretty_generate(formula.to_hash_with_variations)

      File.write("_data/formula/#{name.tr("+", "_")}.json", "#{json}\n")
      File.write("api/formula/#{name}.json", JSON_TEMPLATE)
      File.write("formula/#{name}.html", html_template(name))
    end

    canonical_json = JSON.pretty_generate(tap.formula_renames.merge(tap.alias_table))
    File.write("_data/formula_canonical.json", "#{canonical_json}\n")
  end
end
