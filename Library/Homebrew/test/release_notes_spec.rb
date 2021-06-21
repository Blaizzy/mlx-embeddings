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
      system "git", "commit", "--allow-empty", "-m", "another change"
      system "git", "commit", "--allow-empty", "-m", "Merge pull request #3 from User/another_change"
    end
  end

  describe ".generate_release_notes" do
    it "generates markdown release notes" do
      expect(described_class.generate_release_notes("release-notes-testing", "HEAD")).to eq <<~NOTES
        - [Merge pull request #3 from User/another_change](https://github.com/Homebrew/brew/pull/3) (@User)
        - [Do something else](https://github.com/Homebrew/brew/pull/2) (@User)
      NOTES
    end
  end
end
