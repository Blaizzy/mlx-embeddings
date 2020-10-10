# typed: false
# frozen_string_literal: true

require "livecheck/strategy/git"

describe Homebrew::Livecheck::Strategy::Git do
  subject(:git) { described_class }

  let(:git_url) { "https://github.com/Homebrew/brew.git" }
  let(:non_git_url) { "https://brew.sh/test" }

  describe "::tag_info", :needs_network do
    it "returns the Git tags for the provided remote URL that match the regex provided" do
      expect(git.tag_info(git_url, /^v?(\d+(?:\.\d+))$/))
        .not_to be_empty
    end
  end

  describe "::match?" do
    it "returns true if the argument provided is a Git repository" do
      expect(git.match?(git_url)).to be true
    end

    it "returns false if the argument provided is not a Git repository" do
      expect(git.match?(non_git_url)).to be false
    end
  end
end
