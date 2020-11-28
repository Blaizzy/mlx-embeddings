# typed: false
# frozen_string_literal: true

require "dev-cmd/pr-pull"
require "utils/git"
require "tap"
require "cmd/shared_examples/args_parse"

describe Homebrew do
  let(:formula_rebuild) do
    <<~EOS
      class Foo < Formula
        desc "Helpful description"
        url "https://brew.sh/foo-1.0.tgz"
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
  let(:formula_version) do
    <<~EOS
      class Foo < Formula
        url "https://brew.sh/foo-2.0.tgz"
      end
    EOS
  end
  let(:formula) do
    <<~EOS
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
      end
    EOS
  end
  let(:formula_file) { path/"Formula/foo.rb" }
  let(:path) { (Tap::TAP_DIRECTORY/"homebrew/homebrew-foo").extend(GitRepositoryExtension) }

  describe "Homebrew.pr_pull_args" do
    it_behaves_like "parseable arguments"
  end

  describe "#autosquash!" do
    it "squashes a formula correctly" do
      secondary_author = "Someone Else <me@example.com>"
      (path/"Formula").mkpath
      formula_file.write(formula)
      cd path do
        safe_system Utils::Git.git, "init"
        safe_system Utils::Git.git, "add", formula_file
        safe_system Utils::Git.git, "commit", "-m", "foo 1.0 (new formula)"
        original_hash = `git rev-parse HEAD`.chomp
        File.open(formula_file, "w") { |f| f.write(formula_revision) }
        safe_system Utils::Git.git, "commit", formula_file, "-m", "revision"
        File.open(formula_file, "w") { |f| f.write(formula_version) }
        safe_system Utils::Git.git, "commit", formula_file, "-m", "version", "--author=#{secondary_author}"
        described_class.autosquash!(original_hash, path: path)
        expect(path.git_commit_message).to include("foo 2.0")
        expect(path.git_commit_message).to include("Co-authored-by: #{secondary_author}")
      end
    end
  end

  describe "#signoff!" do
    it "signs off a formula" do
      (path/"Formula").mkpath
      formula_file.write(formula)
      cd path do
        safe_system Utils::Git.git, "init"
        safe_system Utils::Git.git, "add", formula_file
        safe_system Utils::Git.git, "commit", "-m", "foo 1.0 (new formula)"
      end
      described_class.signoff!(path)
      expect(path.git_commit_message).to include("Signed-off-by:")
    end
  end

  describe "#determine_bump_subject" do
    it "correctly bumps a new formula" do
      expect(described_class.determine_bump_subject("", formula, formula_file)).to eq("foo 1.0 (new formula)")
    end

    it "correctly bumps a formula version" do
      expect(described_class.determine_bump_subject(formula, formula_version, formula_file)).to eq("foo 2.0")
    end

    it "correctly bumps a formula revision with reason" do
      expect(described_class.determine_bump_subject(
               formula, formula_revision, formula_file, reason: "for fun"
             )).to eq("foo: revision for fun")
    end

    it "correctly bumps a formula rebuild" do
      expect(described_class.determine_bump_subject(formula, formula_rebuild, formula_file)).to eq("foo: rebuild")
    end

    it "correctly bumps a formula deletion" do
      expect(described_class.determine_bump_subject(formula, "", formula_file)).to eq("foo: delete")
    end
  end
end
