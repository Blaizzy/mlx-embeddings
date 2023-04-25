# typed: true
# frozen_string_literal: true

require "cli/parser"
require "cask/cask"

module Homebrew
  module_function

  sig { returns(CLI::Parser) }
  def generate_cask_api_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Generates Cask API data files for formulae.brew.sh.

        The generated files are written to the current directory.
      EOS

      named_args :none
    end
  end

  CASK_JSON_TEMPLATE = <<~EOS
    ---
    layout: cask_json
    ---
    {{ content }}
  EOS

  def html_template(title)
    <<~EOS
      ---
      title: #{title}
      layout: cask
      ---
      {{ content }}
    EOS
  end

  def generate_cask_api
    generate_cask_api_args.parse

    tap = Tap.default_cask_tap

    directories = ["_data/cask", "api/cask", "api/cask-source", "cask"].freeze
    FileUtils.rm_rf directories
    FileUtils.mkdir_p directories

    Cask::Cask.generating_hash!

    tap.cask_files.each do |path|
      cask = Cask::CaskLoader.load(path)
      name = cask.token
      json = JSON.pretty_generate(cask.to_hash_with_variations)

      File.write("_data/cask/#{name}.json", "#{json}\n")
      File.write("api/cask/#{name}.json", CASK_JSON_TEMPLATE)
      File.write("api/cask-source/#{name}.rb", path.read)
      File.write("cask/#{name}.html", html_template(name))
    rescue
      onoe "Error while generating data for cask '#{path.stem}'."
      raise
    end
  end
end
