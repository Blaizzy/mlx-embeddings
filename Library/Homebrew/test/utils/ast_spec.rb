# typed: false
# frozen_string_literal: true

require "utils/ast"

describe Utils::AST do
  let(:initial_formula) do
    <<~RUBY
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tar.gz"
        license all_of: [
          :public_domain,
          "MIT",
          "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
        ]
      end
    RUBY
  end

  describe ".replace_formula_stanza!" do
    it "replaces the specified stanza in a formula" do
      contents = initial_formula.dup
      described_class.replace_formula_stanza! contents, name: :license, replacement: "license :public_domain"
      expect(contents).to eq <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tar.gz"
          license :public_domain
        end
      RUBY
    end
  end

  describe ".add_formula_stanza!" do
    it "adds the specified stanza to a formula" do
      contents = initial_formula.dup
      described_class.add_formula_stanza! contents, name: :revision, text: "revision 1"
      expect(contents).to eq <<~RUBY
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tar.gz"
          license all_of: [
            :public_domain,
            "MIT",
            "GPL-3.0-or-later" => { with: "Autoconf-exception-3.0" },
          ]
          revision 1
        end
      RUBY
    end
  end
end
