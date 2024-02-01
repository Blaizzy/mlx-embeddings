# typed: strict

module AbstractDownloadStrategy::Pourable
  requires_ancestor { AbstractDownloadStrategy }
end

class Mechanize::HTTP
  ContentDisposition = Struct.new :type, :filename, :creation_date,
    :modification_date, :read_date, :size, :parameters
end

class Mechanize::HTTP::ContentDispositionParser
  sig {
    params(content_disposition: String, header: T::Boolean)
    .returns(T.nilable(Mechanize::HTTP::ContentDisposition))
  }
  def parse(content_disposition, header = false); end
end
