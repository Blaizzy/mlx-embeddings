# frozen_string_literal: true

require "formula_text_auditor"

RSpec.describe Homebrew::FormulaTextAuditor do
  alias_matcher :have_data, :be_data
  alias_matcher :have_end, :be_end
  alias_matcher :have_trailing_newline, :be_trailing_newline

  let(:dir) { mktmpdir }

  def formula_text(name, body = nil, options = {})
    path = dir/"#{name}.rb"

    path.write <<~RUBY
      class #{Formulary.class_s(name)} < Formula
        #{body}
      end
      #{options[:patch]}
    RUBY

    described_class.new(path)
  end

  specify "simple valid Formula" do
    ft = formula_text "valid", <<~RUBY
      url "https://www.brew.sh/valid-1.0.tar.gz"
    RUBY

    expect(ft).to have_trailing_newline

    expect(ft =~ /\burl\b/).to be_truthy
    expect(ft.line_number(/desc/)).to be_nil
    expect(ft.line_number(/\burl\b/)).to eq(2)
    expect(ft).to include("Valid")
  end

  specify "#trailing_newline?" do
    ft = formula_text "newline"
    expect(ft).to have_trailing_newline
  end
end
