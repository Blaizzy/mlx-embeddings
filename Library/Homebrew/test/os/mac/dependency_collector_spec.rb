# frozen_string_literal: true

require "dependency_collector"

describe DependencyCollector do
  alias_matcher :need_tar_xz_dependency, :be_tar_needs_xz_dependency

  specify "Resource dependency from a '.xz' URL" do
    resource = Resource.new
    resource.url("https://brew.sh/foo.tar.xz")
    expect(subject.add(resource)).to be nil
  end

  specify "Resource dependency from a '.zip' URL" do
    resource = Resource.new
    resource.url("https://brew.sh/foo.zip")
    expect(subject.add(resource)).to be nil
  end

  specify "Resource dependency from a '.bz2' URL" do
    resource = Resource.new
    resource.url("https://brew.sh/foo.tar.bz2")
    expect(subject.add(resource)).to be nil
  end

  specify "Resource dependency from a '.git' URL" do
    resource = Resource.new
    resource.url("git://brew.sh/foo/bar.git")
    expect(subject.add(resource)).to be nil
  end

  specify "Resource dependency from a Subversion URL" do
    resource = Resource.new
    resource.url("svn://brew.sh/foo/bar")
    expect(subject.add(resource)).to be nil
  end
end
