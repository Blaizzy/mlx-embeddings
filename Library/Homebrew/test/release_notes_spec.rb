# typed: false
# frozen_string_literal: true

require "release_notes"

describe ReleaseNotes do
  before do
    HOMEBREW_REPOSITORY.cd do
      system "git", "init"
      system "git", "commit", "--allow-empty", "-m", "Initial commit"
      system "git", "tag", "release-notes-testing"
      system "git", "commit", "--allow-empty", "-m", "Merge pull request #1 from Homebrew/fix", "-m", "Do something"
      system "git", "commit", "--allow-empty", "-m", "make a change"
      system "git", "commit", "--allow-empty", "-m", "Merge pull request #2 from User/fix", "-m", "Do something else"
    end
  end

  describe ".generate_release_notes" do
    it "generates release notes" do
      expect(described_class.generate_release_notes("release-notes-testing", "HEAD")).to eq <<~NOTES
        https://github.com/Homebrew/brew/pull/2 (@User) - Do something else
        https://github.com/Homebrew/brew/pull/1 (@Homebrew) - Do something
      NOTES
    end

    it "generates markdown release notes" do
      expect(described_class.generate_release_notes("release-notes-testing", "HEAD", markdown: true)).to eq <<~NOTES
        - [Do something else](https://github.com/Homebrew/brew/pull/2) (@User)
        - [Do something](https://github.com/Homebrew/brew/pull/1) (@Homebrew)
      NOTES
    end
  end
end
