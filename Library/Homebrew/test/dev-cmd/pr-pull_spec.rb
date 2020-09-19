# frozen_string_literal: true

require "dev-cmd/pr-pull"
require "cmd/shared_examples/args_parse"

describe Homebrew do
  describe "Homebrew.pr_pull_args" do
    it_behaves_like "parseable arguments"
  end

  describe "#determine_bump_subject" do
    let(:formula) do
      <<~EOS
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
        end
      EOS
    end

    let(:formula_version) do
      <<~EOS
        class Foo < Formula
          url "https://brew.sh/foo-2.0.tgz"
        end
      EOS
    end

    let(:formula_revision) do
      <<~EOS
        class Foo < Formula
          url "https://brew.sh/foo-1.0.tgz"
          revision 1
        end
      EOS
    end

    let(:formula_rebuild) do
      <<~EOS
        class Foo < Formula
          desc "Helpful description"
          url "https://brew.sh/foo-1.0.tgz"
        end
      EOS
    end

    it "correctly bumps a new formula" do
      expect(described_class.determine_bump_subject("", formula, "foo.rb")).to eq("foo 1.0 (new formula)")
    end

    it "correctly bumps a formula version" do
      expect(described_class.determine_bump_subject(formula, formula_version, "foo.rb")).to eq("foo 2.0")
    end

    it "correctly bumps a formula revision with reason" do
      expect(described_class.determine_bump_subject(
               formula, formula_revision, "foo.rb", reason: "for fun"
             )).to eq("foo: revision for fun")
    end

    it "correctly bumps a formula rebuild" do
      expect(described_class.determine_bump_subject(formula, formula_rebuild, "foo.rb")).to eq("foo: rebuild")
    end

    it "correctly bumps a formula deletion" do
      expect(described_class.determine_bump_subject(formula, "", "foo.rb")).to eq("foo: delete")
    end
  end
end
