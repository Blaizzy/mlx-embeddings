# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::Desc do
  subject(:cop) { described_class.new }

  it "does not start with an indefinite article" do
    expect_no_offenses <<~RUBY
      cask "foo" do
        desc "Bar program"
      end
    RUBY

    expect_offense <<~RUBY, "/homebrew-cask/Casks/foo.rb"
      cask 'foo' do
        desc 'A bar program'
              ^ Description shouldn\'t start with an indefinite article, i.e. \"A\".
      end
    RUBY

    expect_correction <<~RUBY
      cask 'foo' do
        desc 'Bar program'
      end
    RUBY
  end

  it "does not start with the cask name" do
    expect_offense <<~RUBY, "/homebrew-cask/Casks/foo.rb"
      cask 'foobar' do
        desc 'Foo bar program'
              ^^^^^^^ Description shouldn't start with the cask name.
      end
    RUBY

    expect_offense <<~RUBY, "/homebrew-cask/Casks/foo.rb"
      cask 'foobar' do
        desc 'Foo-Bar program'
              ^^^^^^^ Description shouldn\'t start with the cask name.
      end
    RUBY

    expect_offense <<~RUBY, "/homebrew-cask/Casks/foo.rb"
      cask 'foo-bar' do
        desc 'Foo bar program'
              ^^^^^^^ Description shouldn\'t start with the cask name.
      end
    RUBY

    expect_offense <<~RUBY, "/homebrew-cask/Casks/foo.rb"
      cask 'foo-bar' do
        desc 'Foo-Bar program'
              ^^^^^^^ Description shouldn\'t start with the cask name.
      end
    RUBY

    expect_offense <<~RUBY, "/homebrew-cask/Casks/foo.rb"
      cask 'foo-bar' do
        desc 'Foo Bar'
              ^^^^^^^ Description shouldn\'t start with the cask name.
      end
    RUBY
  end
end
