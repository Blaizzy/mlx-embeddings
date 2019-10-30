# frozen_string_literal: true

require "digest"
require "erb"

module Homebrew
  class FormulaCreator
    attr_reader :url, :sha256, :desc, :homepage
    attr_accessor :name, :version, :tap, :path, :mode

    def url=(url)
      @url = url
      path = Pathname.new(url)
      if @name.nil?
        case url
        when %r{github\.com/(\S+)/(\S+)\.git}
          @user = Regexp.last_match(1)
          @name = Regexp.last_match(2)
          @head = true
          @github = true
        when %r{github\.com/(\S+)/(\S+)/(archive|releases)/}
          @user = Regexp.last_match(1)
          @name = Regexp.last_match(2)
          @github = true
        else
          @name = path.basename.to_s[/(.*?)[-_.]?#{Regexp.escape(path.version.to_s)}/, 1]
        end
      end
      update_path
      if @version
        @version = Version.create(@version)
      else
        @version = Version.detect(url, {})
      end
    end

    def update_path
      return if @name.nil? || @tap.nil?

      @path = Formulary.path "#{@tap}/#{@name}"
    end

    def fetch?
      !Homebrew.args.no_fetch?
    end

    def head?
      @head || Homebrew.args.HEAD?
    end

    def generate!
      raise "#{path} already exists" if path.exist?

      if version.nil? || version.null?
        opoo "Version cannot be determined from URL."
        puts "You'll need to add an explicit 'version' to the formula."
      elsif fetch?
        unless head?
          r = Resource.new
          r.url(url)
          r.version(version)
          r.owner = self
          @sha256 = r.fetch.sha256 if r.download_strategy == CurlDownloadStrategy
        end

        if @user && @name
          begin
            metadata = GitHub.repository(@user, @name)
            @desc = metadata["description"]
            @homepage = metadata["homepage"]
          rescue GitHub::HTTPNotFoundError
            # If there was no repository found assume the network connection is at
            # fault rather than the input URL.
            nil
          end
        end
      end

      path.write ERB.new(template, trim_mode: ">").result(binding)
    end

    def template
      <<~ERB
        # Documentation: https://docs.brew.sh/Formula-Cookbook
        #                https://rubydoc.brew.sh/Formula
        # PLEASE REMOVE ALL GENERATED COMMENTS BEFORE SUBMITTING YOUR PULL REQUEST!
        class #{Formulary.class_s(name)} < Formula
        <% if mode == :python %>
          include Language::Python::Virtualenv

        <% end %>
          desc "#{desc}"
          homepage "#{homepage}"
        <% if head? %>
          head "#{url}"
        <% else %>
          url "#{url}"
        <% unless version.nil? or version.detected_from_url? %>
          version "#{version}"
        <% end %>
          sha256 "#{sha256}"
        <% end %>

        <% if mode == :cmake %>
          depends_on "cmake" => :build
        <% elsif mode == :go %>
          depends_on "go" => :build
        <% elsif mode == :meson %>
          depends_on "meson" => :build
          depends_on "ninja" => :build
        <% elsif mode == :perl %>
          uses_from_macos "perl"
        <% elsif mode == :python %>
          depends_on "python"
        <% elsif mode == :rust %>
          depends_on "rust" => :build
        <% elsif mode.nil? %>
          # depends_on "cmake" => :build
        <% end %>

        <% if mode == :perl || mode == :python %>
          # Additional dependency
          # resource "" do
          #   url ""
          #   sha256 ""
          # end

        <% end %>
          def install
            # ENV.deparallelize  # if your formula fails when building in parallel
        <% if mode == :cmake %>
            system "cmake", ".", *std_cmake_args
        <% elsif mode == :autotools %>
            # Remove unrecognized options if warned by configure
            system "./configure", "--disable-debug",
                                  "--disable-dependency-tracking",
                                  "--disable-silent-rules",
                                  "--prefix=\#{prefix}"
        <% elsif mode == :go %>
            system "go", "build", "-o", "\#{bin}/\#{name}"
        <% elsif mode == :meson %>
            mkdir "build" do
              system "meson", "--prefix=\#{prefix}", ".."
              system "ninja", "-v"
              system "ninja", "install", "-v"
            end
        <% elsif mode == :perl %>
            ENV.prepend_create_path "PERL5LIB", libexec/"lib/perl5"
            ENV.prepend_path "PERL5LIB", libexec/"lib"

            # Stage additional dependency (Makefile.PL style)
            # resource("").stage do
            #   system "perl", "Makefile.PL", "INSTALL_BASE=\#{libexec}"
            #   system "make"
            #   system "make", "install"
            # end

            # Stage additional dependency (Build.PL style)
            # resource("").stage do
            #   system "perl", "Build.PL", "--install_base", libexec
            #   system "./Build"
            #   system "./Build", "install"
            # end

            bin.install name
            bin.env_script_all_files(libexec/"bin", :PERL5LIB => ENV["PERL5LIB"])
        <% elsif mode == :python %>
            virtualenv_install_with_resources
        <% elsif mode == :rust %>
            system "cargo", "install", "--locked", "--root", prefix, "--path", "."
        <% else %>
            # Remove unrecognized options if warned by configure
            system "./configure", "--disable-debug",
                                  "--disable-dependency-tracking",
                                  "--disable-silent-rules",
                                  "--prefix=\#{prefix}"
            # system "cmake", ".", *std_cmake_args
        <% end %>
        <% if mode == :autotools || mode == :cmake %>
            system "make", "install" # if this fails, try separate make/make install steps
        <% end %>
          end

          test do
            # `test do` will create, run in and delete a temporary directory.
            #
            # This test will fail and we won't accept that! For Homebrew/homebrew-core
            # this will need to be a test that verifies the functionality of the
            # software. Run the test with `brew test #{name}`. Options passed
            # to `brew install` such as `--HEAD` also need to be provided to `brew test`.
            #
            # The installed folder is not in the path, so use the entire path to any
            # executables being tested: `system "\#{bin}/program", "do", "something"`.
            system "false"
          end
        end
      ERB
    end
  end
end
